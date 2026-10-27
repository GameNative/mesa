#include <string.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "vk_enum_to_str.h"
#include "vk_util.h"
#include "util/ralloc.h"
#include "util/hash_table.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/u_math.h"

#define WRAPPER_STRIDE_MAX_BINDINGS 32

struct wrapper_stride_code {
   uint32_t *code;
   size_t size;
};

struct wrapper_variant_key {
   uint32_t strides[WRAPPER_STRIDE_MAX_BINDINGS];
   uint32_t eds[WRAPPER_EDS_COUNT];
};

struct wrapper_stride_variant {
   struct wrapper_variant_key key;
   VkPipeline pipeline;
};

struct wrapper_stride_pipeline {
   void *mem;
   simple_mtx_t mtx;
   VkPipelineCache cache;
   VkGraphicsPipelineCreateInfo info;
   VkPipelineVertexInputStateCreateInfo vi;
   VkVertexInputBindingDescription *bindings;
   VkPipelineInputAssemblyStateCreateInfo *ia;
   VkPipelineRasterizationStateCreateInfo *rs;
   VkPipelineDepthStencilStateCreateInfo *ds;
   VkPipelineViewportStateCreateInfo *vp;
   uint32_t eds_dynamic;
   bool stride_dynamic;
   VkShaderModule *modules;
   uint32_t module_count;
   uint32_t variant_count;
   uint32_t variant_capacity;
   struct wrapper_stride_variant *variants;
};

void
wrapper_stride_note_module(struct wrapper_device *device, VkShaderModule module,
                           const uint32_t *code, size_t size)
{
   struct wrapper_stride_code *c = malloc(sizeof(*c));
   if (!c)
      return;
   c->code = malloc(size);
   if (!c->code) {
      free(c);
      return;
   }
   memcpy(c->code, code, size);
   c->size = size;
   simple_mtx_lock(&device->resource_mutex);
   _mesa_hash_table_u64_insert(device->shader_code_table, (uint64_t)module, c);
   simple_mtx_unlock(&device->resource_mutex);
}

uint32_t *
wrapper_stride_module_code(struct wrapper_device *device, VkShaderModule module, size_t *size)
{
   uint32_t *copy = NULL;
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_stride_code *c =
      _mesa_hash_table_u64_search(device->shader_code_table, (uint64_t)module);
   if (c && (copy = malloc(c->size))) {
      memcpy(copy, c->code, c->size);
      *size = c->size;
   }
   simple_mtx_unlock(&device->resource_mutex);
   return copy;
}

void
wrapper_stride_forget_module(struct wrapper_device *device, VkShaderModule module)
{
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_stride_code *c =
      _mesa_hash_table_u64_search(device->shader_code_table, (uint64_t)module);
   if (c)
      _mesa_hash_table_u64_remove(device->shader_code_table, (uint64_t)module);
   simple_mtx_unlock(&device->resource_mutex);
   if (c) {
      free(c->code);
      free(c);
   }
}

static bool
has_dynamic_stride(const VkPipelineDynamicStateCreateInfo *dyn)
{
   if (!dyn)
      return false;
   for (uint32_t i = 0; i < dyn->dynamicStateCount; i++)
      if (dyn->pDynamicStates[i] == VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE)
         return true;
   return false;
}

static uint32_t
eds_state_mask(const struct wrapper_device *device, VkDynamicState state)
{
   if (device->emulate_eds) {
      switch (state) {
      case VK_DYNAMIC_STATE_CULL_MODE:
         return BITFIELD_BIT(WRAPPER_EDS_CULL);
      case VK_DYNAMIC_STATE_FRONT_FACE:
         return BITFIELD_BIT(WRAPPER_EDS_FRONT);
      case VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY:
         return BITFIELD_BIT(WRAPPER_EDS_TOPOLOGY);
      case VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT:
         return BITFIELD_BIT(WRAPPER_EDS_VIEWPORTS);
      case VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT:
         return BITFIELD_BIT(WRAPPER_EDS_SCISSORS);
      case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_DEPTH_TEST);
      case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_DEPTH_WRITE);
      case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP:
         return BITFIELD_BIT(WRAPPER_EDS_DEPTH_COMPARE);
      case VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_DEPTH_BOUNDS);
      case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_STENCIL_TEST);
      case VK_DYNAMIC_STATE_STENCIL_OP:
         return BITFIELD_RANGE(WRAPPER_EDS_FRONT_FAIL, 8);
      default:
         break;
      }
   }
   if (device->emulate_eds2) {
      switch (state) {
      case VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_RASTER_DISCARD);
      case VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_DEPTH_BIAS);
      case VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE:
         return BITFIELD_BIT(WRAPPER_EDS_PRIM_RESTART);
      default:
         break;
      }
   }
   return 0;
}

static uint32_t
eds_dynamic_mask(const struct wrapper_device *device,
                 const VkPipelineDynamicStateCreateInfo *dyn)
{
   uint32_t mask = 0;
   if (!dyn)
      return 0;
   for (uint32_t i = 0; i < dyn->dynamicStateCount; i++)
      mask |= eds_state_mask(device, dyn->pDynamicStates[i]);
   return mask;
}

bool
wrapper_stride_wants(const struct wrapper_device *device,
                     const VkGraphicsPipelineCreateInfo *info)
{
   if (!device->emulate_dynamic_stride && !device->emulate_eds && !device->emulate_eds2)
      return false;
   if ((info->flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) ||
       vk_find_struct_const(info->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR) ||
       vk_find_struct_const(info->pNext, GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT) ||
       !info->pVertexInputState ||
       info->pVertexInputState->vertexBindingDescriptionCount > WRAPPER_STRIDE_MAX_BINDINGS)
      return false;
   return (device->emulate_dynamic_stride && has_dynamic_stride(info->pDynamicState)) ||
          eds_dynamic_mask(device, info->pDynamicState);
}

static void *
copy_array(void *mem, const void *src, size_t size)
{
   return src && size ? ralloc_memdup(mem, src, size) : NULL;
}

static const void *
copy_chain(void *mem, const void *pNext)
{
   const VkBaseInStructure *in = pNext;
   VkBaseOutStructure *head = NULL, *tail = NULL;

   for (; in; in = in->pNext) {
      switch (in->sType) {
      case VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO:
      case VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO:
      case VK_STRUCTURE_TYPE_PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT:
         continue;
      case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO:
      case VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR:
      case VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO:
      case VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO:
      case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_KHR:
      case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT:
      case VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO:
      case VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO:
         break;
      default:
         WRAPPER_LOG(info, "Dynamic stride emulation drops pipeline struct %d", in->sType);
         continue;
      }

      VkBaseOutStructure *out = ralloc_memdup(mem, in, vk_structure_type_size(in));
      out->pNext = NULL;

      switch (in->sType) {
      case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: {
         VkPipelineRenderingCreateInfo *r = (void *)out;
         r->pColorAttachmentFormats = copy_array(mem, r->pColorAttachmentFormats,
            r->colorAttachmentCount * sizeof(VkFormat));
         break;
      }
      case VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR: {
         VkPipelineVertexInputDivisorStateCreateInfoKHR *d = (void *)out;
         d->pVertexBindingDivisors = copy_array(mem, d->pVertexBindingDivisors,
            d->vertexBindingDivisorCount * sizeof(VkVertexInputBindingDivisorDescriptionKHR));
         break;
      }
      case VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT: {
         VkPipelineColorWriteCreateInfoEXT *w = (void *)out;
         w->pColorWriteEnables = copy_array(mem, w->pColorWriteEnables,
            w->attachmentCount * sizeof(VkBool32));
         break;
      }
      default:
         break;
      }

      if (tail)
         tail->pNext = out;
      else
         head = out;
      tail = out;
   }
   return head;
}

#define COPY_STATE(field, type)                                      \
   if (src->field) {                                                 \
      type *s = ralloc_memdup(mem, src->field, sizeof(type));        \
      s->pNext = copy_chain(mem, s->pNext);                          \
      dst->field = s;                                                \
   }

static bool
copy_create_info(struct wrapper_device *device, struct wrapper_stride_pipeline *sp,
                 const VkGraphicsPipelineCreateInfo *src)
{
   void *mem = sp->mem;
   VkGraphicsPipelineCreateInfo *dst = &sp->info;

   *dst = *src;
   dst->pNext = copy_chain(mem, src->pNext);
   dst->basePipelineHandle = VK_NULL_HANDLE;
   dst->basePipelineIndex = -1;
   dst->flags &= ~VK_PIPELINE_CREATE_DERIVATIVE_BIT;

   VkPipelineShaderStageCreateInfo *stages =
      ralloc_array(mem, VkPipelineShaderStageCreateInfo, src->stageCount ? src->stageCount : 1);
   sp->modules = ralloc_array(mem, VkShaderModule, src->stageCount ? src->stageCount : 1);
   for (uint32_t i = 0; i < src->stageCount; i++) {
      stages[i] = src->pStages[i];
      stages[i].pNext = copy_chain(mem, stages[i].pNext);
      stages[i].pName = ralloc_strdup(mem, src->pStages[i].pName);
      if (src->pStages[i].pSpecializationInfo) {
         const VkSpecializationInfo *si = src->pStages[i].pSpecializationInfo;
         VkSpecializationInfo *s = ralloc_memdup(mem, si, sizeof(*si));
         s->pMapEntries = copy_array(mem, si->pMapEntries,
            si->mapEntryCount * sizeof(VkSpecializationMapEntry));
         s->pData = copy_array(mem, si->pData, si->dataSize);
         stages[i].pSpecializationInfo = s;
      }

      simple_mtx_lock(&device->resource_mutex);
      struct wrapper_stride_code *c = _mesa_hash_table_u64_search(
         device->shader_code_table, (uint64_t)src->pStages[i].module);
      VkShaderModuleCreateInfo mi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = c ? c->size : 0,
         .pCode = c ? c->code : NULL,
      };
      VkResult r = c ? device->dispatch_table.CreateShaderModule(device->dispatch_handle,
                                                                 &mi, NULL, &sp->modules[i])
                     : VK_ERROR_UNKNOWN;
      simple_mtx_unlock(&device->resource_mutex);
      if (r != VK_SUCCESS)
         return false;
      sp->module_count = i + 1;
      stages[i].module = sp->modules[i];
   }
   dst->pStages = stages;

   sp->vi = *src->pVertexInputState;
   sp->vi.pNext = copy_chain(mem, sp->vi.pNext);
   sp->bindings = ralloc_array(mem, VkVertexInputBindingDescription, WRAPPER_STRIDE_MAX_BINDINGS);
   memcpy(sp->bindings, sp->vi.pVertexBindingDescriptions,
          sp->vi.vertexBindingDescriptionCount * sizeof(VkVertexInputBindingDescription));
   sp->vi.pVertexBindingDescriptions = sp->bindings;
   sp->vi.pVertexAttributeDescriptions = copy_array(mem, sp->vi.pVertexAttributeDescriptions,
      sp->vi.vertexAttributeDescriptionCount * sizeof(VkVertexInputAttributeDescription));
   dst->pVertexInputState = &sp->vi;

   COPY_STATE(pInputAssemblyState, VkPipelineInputAssemblyStateCreateInfo);
   COPY_STATE(pTessellationState, VkPipelineTessellationStateCreateInfo);
   if (src->pViewportState) {
      VkPipelineViewportStateCreateInfo *v = ralloc_memdup(mem, src->pViewportState, sizeof(*v));
      v->pNext = copy_chain(mem, v->pNext);
      v->pViewports = copy_array(mem, v->pViewports, v->viewportCount * sizeof(VkViewport));
      v->pScissors = copy_array(mem, v->pScissors, v->scissorCount * sizeof(VkRect2D));
      dst->pViewportState = v;
   } else if (sp->eds_dynamic & (BITFIELD_BIT(WRAPPER_EDS_VIEWPORTS) |
                                 BITFIELD_BIT(WRAPPER_EDS_SCISSORS))) {
      VkPipelineViewportStateCreateInfo *v = rzalloc(mem, VkPipelineViewportStateCreateInfo);
      v->sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      dst->pViewportState = v;
   }
   COPY_STATE(pRasterizationState, VkPipelineRasterizationStateCreateInfo);
   if (src->pMultisampleState) {
      VkPipelineMultisampleStateCreateInfo *m = ralloc_memdup(mem, src->pMultisampleState, sizeof(*m));
      m->pNext = copy_chain(mem, m->pNext);
      m->pSampleMask = copy_array(mem, m->pSampleMask,
         DIV_ROUND_UP(m->rasterizationSamples, 32) * sizeof(VkSampleMask));
      dst->pMultisampleState = m;
   }
   COPY_STATE(pDepthStencilState, VkPipelineDepthStencilStateCreateInfo);
   if (src->pColorBlendState) {
      VkPipelineColorBlendStateCreateInfo *b = ralloc_memdup(mem, src->pColorBlendState, sizeof(*b));
      b->pNext = copy_chain(mem, b->pNext);
      b->pAttachments = copy_array(mem, b->pAttachments,
         b->attachmentCount * sizeof(VkPipelineColorBlendAttachmentState));
      dst->pColorBlendState = b;
   }

   const VkPipelineDynamicStateCreateInfo *dyn = src->pDynamicState;
   VkPipelineDynamicStateCreateInfo *d = ralloc_memdup(mem, dyn, sizeof(*d));
   VkDynamicState *states = ralloc_array(mem, VkDynamicState, dyn->dynamicStateCount + 2);
   bool has_viewport = false, has_scissor = false;
   d->pNext = NULL;
   d->dynamicStateCount = 0;
   for (uint32_t i = 0; i < dyn->dynamicStateCount; i++) {
      VkDynamicState state = dyn->pDynamicStates[i];
      if (state == VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE && sp->stride_dynamic)
         continue;
      if (state == VK_DYNAMIC_STATE_VIEWPORT)
         has_viewport = true;
      if (state == VK_DYNAMIC_STATE_SCISSOR)
         has_scissor = true;
      if (eds_state_mask(device, state))
         continue;
      states[d->dynamicStateCount++] = state;
   }
   if ((sp->eds_dynamic & BITFIELD_BIT(WRAPPER_EDS_VIEWPORTS)) && !has_viewport)
      states[d->dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
   if ((sp->eds_dynamic & BITFIELD_BIT(WRAPPER_EDS_SCISSORS)) && !has_scissor)
      states[d->dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;
   d->pDynamicStates = states;
   dst->pDynamicState = d;

   sp->ia = (VkPipelineInputAssemblyStateCreateInfo *)dst->pInputAssemblyState;
   sp->rs = (VkPipelineRasterizationStateCreateInfo *)dst->pRasterizationState;
   sp->ds = (VkPipelineDepthStencilStateCreateInfo *)dst->pDepthStencilState;
   sp->vp = (VkPipelineViewportStateCreateInfo *)dst->pViewportState;
   return true;
}

static void
base_key(const struct wrapper_stride_pipeline *sp, struct wrapper_variant_key *key)
{
   memset(key, 0, sizeof(*key));
   for (uint32_t b = 0; b < sp->vi.vertexBindingDescriptionCount; b++)
      key->strides[sp->bindings[b].binding % WRAPPER_STRIDE_MAX_BINDINGS] = sp->bindings[b].stride;
   uint32_t *e = key->eds;
   if (sp->rs) {
      e[WRAPPER_EDS_CULL] = sp->rs->cullMode;
      e[WRAPPER_EDS_FRONT] = sp->rs->frontFace;
      e[WRAPPER_EDS_RASTER_DISCARD] = sp->rs->rasterizerDiscardEnable;
      e[WRAPPER_EDS_DEPTH_BIAS] = sp->rs->depthBiasEnable;
   }
   if (sp->ia) {
      e[WRAPPER_EDS_TOPOLOGY] = sp->ia->topology;
      e[WRAPPER_EDS_PRIM_RESTART] = sp->ia->primitiveRestartEnable;
   }
   if (sp->ds) {
      e[WRAPPER_EDS_DEPTH_TEST] = sp->ds->depthTestEnable;
      e[WRAPPER_EDS_DEPTH_WRITE] = sp->ds->depthWriteEnable;
      e[WRAPPER_EDS_DEPTH_COMPARE] = sp->ds->depthCompareOp;
      e[WRAPPER_EDS_DEPTH_BOUNDS] = sp->ds->depthBoundsTestEnable;
      e[WRAPPER_EDS_STENCIL_TEST] = sp->ds->stencilTestEnable;
      e[WRAPPER_EDS_FRONT_FAIL] = sp->ds->front.failOp;
      e[WRAPPER_EDS_FRONT_PASS] = sp->ds->front.passOp;
      e[WRAPPER_EDS_FRONT_DEPTH_FAIL] = sp->ds->front.depthFailOp;
      e[WRAPPER_EDS_FRONT_COMPARE] = sp->ds->front.compareOp;
      e[WRAPPER_EDS_BACK_FAIL] = sp->ds->back.failOp;
      e[WRAPPER_EDS_BACK_PASS] = sp->ds->back.passOp;
      e[WRAPPER_EDS_BACK_DEPTH_FAIL] = sp->ds->back.depthFailOp;
      e[WRAPPER_EDS_BACK_COMPARE] = sp->ds->back.compareOp;
   }
   if (sp->vp) {
      e[WRAPPER_EDS_VIEWPORTS] = MAX2(sp->vp->viewportCount, 1);
      e[WRAPPER_EDS_SCISSORS] = MAX2(sp->vp->scissorCount, 1);
   }
}

static void
apply_key(struct wrapper_stride_pipeline *sp, const struct wrapper_variant_key *key)
{
   const uint32_t *e = key->eds;
   for (uint32_t b = 0; b < sp->vi.vertexBindingDescriptionCount; b++)
      sp->bindings[b].stride = key->strides[sp->bindings[b].binding % WRAPPER_STRIDE_MAX_BINDINGS];
   if (sp->rs) {
      sp->rs->cullMode = e[WRAPPER_EDS_CULL];
      sp->rs->frontFace = e[WRAPPER_EDS_FRONT];
      sp->rs->rasterizerDiscardEnable = e[WRAPPER_EDS_RASTER_DISCARD];
      sp->rs->depthBiasEnable = e[WRAPPER_EDS_DEPTH_BIAS];
   }
   if (sp->ia) {
      sp->ia->topology = e[WRAPPER_EDS_TOPOLOGY];
      sp->ia->primitiveRestartEnable = e[WRAPPER_EDS_PRIM_RESTART];
   }
   if (sp->ds) {
      sp->ds->depthTestEnable = e[WRAPPER_EDS_DEPTH_TEST];
      sp->ds->depthWriteEnable = e[WRAPPER_EDS_DEPTH_WRITE];
      sp->ds->depthCompareOp = e[WRAPPER_EDS_DEPTH_COMPARE];
      sp->ds->depthBoundsTestEnable = e[WRAPPER_EDS_DEPTH_BOUNDS];
      sp->ds->stencilTestEnable = e[WRAPPER_EDS_STENCIL_TEST];
      sp->ds->front.failOp = e[WRAPPER_EDS_FRONT_FAIL];
      sp->ds->front.passOp = e[WRAPPER_EDS_FRONT_PASS];
      sp->ds->front.depthFailOp = e[WRAPPER_EDS_FRONT_DEPTH_FAIL];
      sp->ds->front.compareOp = e[WRAPPER_EDS_FRONT_COMPARE];
      sp->ds->back.failOp = e[WRAPPER_EDS_BACK_FAIL];
      sp->ds->back.passOp = e[WRAPPER_EDS_BACK_PASS];
      sp->ds->back.depthFailOp = e[WRAPPER_EDS_BACK_DEPTH_FAIL];
      sp->ds->back.compareOp = e[WRAPPER_EDS_BACK_COMPARE];
   }
   if (sp->vp && (sp->eds_dynamic & BITFIELD_BIT(WRAPPER_EDS_VIEWPORTS))) {
      sp->vp->viewportCount = e[WRAPPER_EDS_VIEWPORTS];
      sp->vp->pViewports = NULL;
   }
   if (sp->vp && (sp->eds_dynamic & BITFIELD_BIT(WRAPPER_EDS_SCISSORS))) {
      sp->vp->scissorCount = e[WRAPPER_EDS_SCISSORS];
      sp->vp->pScissors = NULL;
   }
}

static void
free_record(struct wrapper_device *device, struct wrapper_stride_pipeline *sp)
{
   for (uint32_t i = 0; i < sp->module_count; i++)
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, sp->modules[i], NULL);
   simple_mtx_destroy(&sp->mtx);
   ralloc_free(sp->mem);
   free(sp);
}

static VkPipeline
create_variant(struct wrapper_device *device, struct wrapper_stride_pipeline *sp,
               const struct wrapper_variant_key *key)
{
   for (uint32_t i = 0; i < sp->variant_count; i++)
      if (!memcmp(&sp->variants[i].key, key, sizeof(*key)))
         return sp->variants[i].pipeline;

   apply_key(sp, key);

   VkPipeline pipeline = VK_NULL_HANDLE;
   if (device->dispatch_table.CreateGraphicsPipelines(device->dispatch_handle, sp->cache, 1,
                                                      &sp->info, NULL, &pipeline) != VK_SUCCESS)
      return VK_NULL_HANDLE;

   if (sp->variant_count == sp->variant_capacity) {
      uint32_t cap = sp->variant_capacity ? sp->variant_capacity * 2 : 4;
      struct wrapper_stride_variant *v =
         reralloc(sp->mem, sp->variants, struct wrapper_stride_variant, cap);
      if (!v)
         return pipeline;
      sp->variants = v;
      sp->variant_capacity = cap;
   }
   sp->variants[sp->variant_count].key = *key;
   sp->variants[sp->variant_count++].pipeline = pipeline;
   return pipeline;
}

VkResult
wrapper_stride_create(struct wrapper_device *device, VkPipelineCache cache,
                      const VkGraphicsPipelineCreateInfo *info,
                      const VkAllocationCallbacks *alloc, VkPipeline *pipeline)
{
   struct wrapper_stride_pipeline *sp = calloc(1, sizeof(*sp));
   if (!sp)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   sp->mem = ralloc_context(NULL);
   sp->cache = cache;
   sp->stride_dynamic = device->emulate_dynamic_stride && has_dynamic_stride(info->pDynamicState);
   sp->eds_dynamic = eds_dynamic_mask(device, info->pDynamicState);
   simple_mtx_init(&sp->mtx, mtx_plain);

   if (!copy_create_info(device, sp, info)) {
      free_record(device, sp);
      return device->dispatch_table.CreateGraphicsPipelines(device->dispatch_handle, cache, 1,
                                                            info, alloc, pipeline);
   }

   struct wrapper_variant_key key;
   base_key(sp, &key);
   *pipeline = create_variant(device, sp, &key);
   sp->cache = VK_NULL_HANDLE;
   if (*pipeline == VK_NULL_HANDLE) {
      free_record(device, sp);
      return VK_ERROR_UNKNOWN;
   }

   simple_mtx_lock(&device->resource_mutex);
   _mesa_hash_table_u64_insert(device->stride_pipeline_table, (uint64_t)*pipeline, sp);
   simple_mtx_unlock(&device->resource_mutex);
   return VK_SUCCESS;
}

void
wrapper_stride_destroy(struct wrapper_device *device, VkPipeline pipeline)
{
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_stride_pipeline *sp =
      _mesa_hash_table_u64_search(device->stride_pipeline_table, (uint64_t)pipeline);
   if (sp)
      _mesa_hash_table_u64_remove(device->stride_pipeline_table, (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);
   if (!sp)
      return;

   for (uint32_t i = 0; i < sp->variant_count; i++)
      if (sp->variants[i].pipeline != pipeline)
         device->dispatch_table.DestroyPipeline(device->dispatch_handle,
                                                sp->variants[i].pipeline, NULL);
   free_record(device, sp);
}

void
wrapper_stride_bind(struct wrapper_command_buffer *wcb, VkPipeline pipeline)
{
   struct wrapper_device *device = wcb->device;
   simple_mtx_lock(&device->resource_mutex);
   wcb->stride_pipeline =
      _mesa_hash_table_u64_search(device->stride_pipeline_table, (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);
   wcb->stride_bound = pipeline;
}

void
wrapper_stride_set(struct wrapper_command_buffer *wcb, uint32_t first, uint32_t count,
                   const VkDeviceSize *strides)
{
   for (uint32_t i = 0; i < count && first + i < WRAPPER_STRIDE_MAX_BINDINGS; i++) {
      wcb->vertex_strides[first + i] = (uint32_t)strides[i];
      wcb->vertex_strides_valid |= 1u << (first + i);
   }
}

void
wrapper_stride_flush(struct wrapper_command_buffer *wcb)
{
   struct wrapper_stride_pipeline *sp = wcb->stride_pipeline;
   if (!sp)
      return;

   simple_mtx_lock(&sp->mtx);
   struct wrapper_variant_key key = sp->variants[0].key;
   if (sp->stride_dynamic) {
      for (uint32_t b = 0; b < sp->vi.vertexBindingDescriptionCount; b++) {
         uint32_t n = sp->bindings[b].binding % WRAPPER_STRIDE_MAX_BINDINGS;
         if (wcb->vertex_strides_valid & (1u << n))
            key.strides[n] = wcb->vertex_strides[n];
      }
   }
   u_foreach_bit(f, sp->eds_dynamic & wcb->eds_valid)
      key.eds[f] = wcb->eds[f];
   VkPipeline pipeline = create_variant(wcb->device, sp, &key);
   simple_mtx_unlock(&sp->mtx);

   if (pipeline != VK_NULL_HANDLE && pipeline != wcb->stride_bound) {
      wcb->device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      wcb->stride_bound = pipeline;
   }
}

void
wrapper_stride_reset(struct wrapper_command_buffer *wcb)
{
   wcb->stride_pipeline = NULL;
   wcb->stride_bound = VK_NULL_HANDLE;
   wcb->vertex_strides_valid = 0;
   wcb->eds_valid = 0;
}

void
wrapper_eds_set(struct wrapper_command_buffer *wcb, enum wrapper_eds_field field,
                uint32_t value)
{
   wcb->eds[field] = value;
   wcb->eds_valid |= BITFIELD_BIT(field);
}

void
wrapper_stride_finish(struct wrapper_device *device)
{
   hash_table_u64_foreach(device->stride_pipeline_table, entry) {
      struct wrapper_stride_pipeline *sp = entry.data;
      for (uint32_t i = 0; i < sp->variant_count; i++)
         device->dispatch_table.DestroyPipeline(device->dispatch_handle,
                                                sp->variants[i].pipeline, NULL);
      free_record(device, sp);
   }
   _mesa_hash_table_u64_destroy(device->stride_pipeline_table);
   hash_table_u64_foreach(device->shader_code_table, entry) {
      struct wrapper_stride_code *c = entry.data;
      free(c->code);
      free(c);
   }
   _mesa_hash_table_u64_destroy(device->shader_code_table);
}
