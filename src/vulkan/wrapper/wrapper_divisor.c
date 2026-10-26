#include "wrapper_private.h"
#include "wrapper_log.h"

#include "util/hash_table.h"
#include "vk_util.h"

#define WRAPPER_DIVISOR_MAX_BINDINGS 32

struct wrapper_divisor_pipeline {
   VkShaderModule module;
   uint32_t mask;
   bool dynamic_stride;
   uint32_t divisor[WRAPPER_DIVISOR_MAX_BINDINGS];
   uint32_t stride[WRAPPER_DIVISOR_MAX_BINDINGS];
};

struct wrapper_divisor_lowering {
   VkPipelineShaderStageCreateInfo stages[8];
   VkPipelineVertexInputStateCreateInfo vi;
   VkVertexInputBindingDescription bindings[WRAPPER_DIVISOR_MAX_BINDINGS];
   VkPipelineVertexInputDivisorStateCreateInfoKHR ds;
   VkVertexInputBindingDivisorDescriptionKHR divisors[WRAPPER_DIVISOR_MAX_BINDINGS];
   struct wrapper_divisor_pipeline info;
};

static const VkPipelineVertexInputDivisorStateCreateInfoKHR *
find_divisor_state(const VkPipelineVertexInputStateCreateInfo *vi)
{
   return vk_find_struct_const(vi->pNext, PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR);
}

static bool
has_dynamic(const VkGraphicsPipelineCreateInfo *info, VkDynamicState state)
{
   if (!info->pDynamicState)
      return false;
   for (uint32_t i = 0; i < info->pDynamicState->dynamicStateCount; i++)
      if (info->pDynamicState->pDynamicStates[i] == state)
         return true;
   return false;
}

#define SPV_OP_TYPE_INT 21
#define SPV_OP_CONSTANT 43
#define SPV_OP_VARIABLE 59
#define SPV_OP_LOAD 61
#define SPV_OP_DECORATE 71
#define SPV_OP_COPY_OBJECT 83
#define SPV_DECORATION_BUILTIN 11
#define SPV_BUILTIN_BASE_INSTANCE 4425

static uint32_t *
zero_base_instance(const uint32_t *code, size_t size, size_t *out_size)
{
   size_t words = size / 4;
   if (words < 5)
      return NULL;
   uint32_t var = 0;
   for (size_t o = 5; o < words;) {
      uint32_t len = code[o] >> 16, op = code[o] & 0xffff;
      if (!len || o + len > words)
         return NULL;
      if (op == SPV_OP_DECORATE && len >= 4 && code[o + 2] == SPV_DECORATION_BUILTIN &&
          code[o + 3] == SPV_BUILTIN_BASE_INSTANCE)
         var = code[o + 1];
      o += len;
   }
   if (!var)
      return NULL;

   uint32_t type = 0;
   size_t type_end = 0;
   for (size_t o = 5; o < words;) {
      uint32_t len = code[o] >> 16, op = code[o] & 0xffff;
      if (op == SPV_OP_LOAD && len >= 4 && code[o + 3] == var)
         type = code[o + 1];
      o += len;
   }
   if (!type)
      return NULL;
   for (size_t o = 5; o < words;) {
      uint32_t len = code[o] >> 16, op = code[o] & 0xffff;
      if (op == SPV_OP_TYPE_INT && code[o + 1] == type)
         type_end = o + len;
      o += len;
   }
   if (!type_end)
      return NULL;

   uint32_t *out = malloc((words + 4) * 4);
   if (!out)
      return NULL;
   uint32_t zero = code[3];
   memcpy(out, code, type_end * 4);
   out[3] = zero + 1;
   out[type_end] = (4u << 16) | SPV_OP_CONSTANT;
   out[type_end + 1] = type;
   out[type_end + 2] = zero;
   out[type_end + 3] = 0;
   memcpy(out + type_end + 4, code + type_end, (words - type_end) * 4);

   for (size_t o = 5; o < words + 4;) {
      uint32_t len = out[o] >> 16, op = out[o] & 0xffff;
      if (op == SPV_OP_LOAD && len >= 4 && out[o + 3] == var) {
         out[o] = (4u << 16) | SPV_OP_COPY_OBJECT;
         out[o + 3] = zero;
         for (uint32_t k = 4; k < len; k++)
            out[o + k] = 1u << 16;
      }
      o += len;
   }
   *out_size = (words + 4) * 4;
   return out;
}

static VkShaderModule
vertex_module_without_base_instance(struct wrapper_device *device, VkShaderModule module)
{
   size_t size = 0, patched_size = 0;
   uint32_t *code = wrapper_stride_module_code(device, module, &size);
   if (!code)
      return VK_NULL_HANDLE;
   uint32_t *patched = zero_base_instance(code, size, &patched_size);
   free(code);
   if (!patched)
      return VK_NULL_HANDLE;
   VkShaderModuleCreateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = patched_size,
      .pCode = patched,
   };
   VkShaderModule out = VK_NULL_HANDLE;
   if (device->dispatch_table.CreateShaderModule(device->dispatch_handle, &mi, NULL, &out) ==
       VK_SUCCESS)
      wrapper_stride_note_module(device, out, patched, patched_size);
   free(patched);
   return out;
}

void *
wrapper_divisor_lower(struct wrapper_device *device,
                      const VkGraphicsPipelineCreateInfo *in,
                      VkGraphicsPipelineCreateInfo *out)
{
   *out = *in;
   bool zero_only = !device->emulate_vertex_divisor;
   if ((zero_only && !device->emulate_zero_divisor) || !in->pVertexInputState ||
       has_dynamic(in, VK_DYNAMIC_STATE_VERTEX_INPUT_EXT))
      return NULL;

   const VkPipelineVertexInputStateCreateInfo *vi = in->pVertexInputState;
   const VkPipelineVertexInputDivisorStateCreateInfoKHR *ds = find_divisor_state(vi);
   if (!ds || vi->vertexBindingDescriptionCount > WRAPPER_DIVISOR_MAX_BINDINGS ||
       ds->vertexBindingDivisorCount > WRAPPER_DIVISOR_MAX_BINDINGS)
      return NULL;
   if (zero_only) {
      bool any_zero = false;
      for (uint32_t i = 0; i < ds->vertexBindingDivisorCount; i++)
         any_zero |= ds->pVertexBindingDivisors[i].divisor == 0;
      if (!any_zero || vi->pNext != (const void *)ds)
         return NULL;
   }

   struct wrapper_divisor_lowering *l = calloc(1, sizeof(*l));
   if (!l)
      return NULL;
   l->vi = *vi;
   memcpy(l->bindings, vi->pVertexBindingDescriptions,
          vi->vertexBindingDescriptionCount * sizeof(*l->bindings));
   l->vi.pVertexBindingDescriptions = l->bindings;
   if (vi->pNext == (const void *)ds)
      l->vi.pNext = ds->pNext;
   l->info.dynamic_stride = has_dynamic(in, VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);

   for (uint32_t i = 0; i < ds->vertexBindingDivisorCount; i++) {
      const VkVertexInputBindingDivisorDescriptionKHR *d = &ds->pVertexBindingDivisors[i];
      if (zero_only && d->divisor != 0) {
         l->divisors[l->ds.vertexBindingDivisorCount++] = *d;
         continue;
      }
      if (d->divisor == 1 || d->binding >= WRAPPER_DIVISOR_MAX_BINDINGS)
         continue;
      for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++) {
         if (l->bindings[b].binding != d->binding ||
             l->bindings[b].inputRate != VK_VERTEX_INPUT_RATE_INSTANCE)
            continue;
         l->info.mask |= 1u << d->binding;
         l->info.divisor[d->binding] = d->divisor;
         l->info.stride[d->binding] = l->bindings[b].stride;
         l->bindings[b].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
         l->bindings[b].stride = 0;
      }
   }
   if (l->ds.vertexBindingDivisorCount) {
      l->ds.sType = ds->sType;
      l->ds.pNext = ds->pNext;
      l->ds.pVertexBindingDivisors = l->divisors;
      l->vi.pNext = &l->ds;
   }
   out->pVertexInputState = &l->vi;

   bool grouped = false;
   u_foreach_bit(b, l->info.mask)
      grouped |= l->info.divisor[b] > 1;
   if (grouped && in->stageCount <= ARRAY_SIZE(l->stages)) {
      memcpy(l->stages, in->pStages, in->stageCount * sizeof(*l->stages));
      for (uint32_t i = 0; i < in->stageCount; i++) {
         if (l->stages[i].stage != VK_SHADER_STAGE_VERTEX_BIT || !l->stages[i].module)
            continue;
         l->info.module = vertex_module_without_base_instance(device, l->stages[i].module);
         if (l->info.module) {
            l->stages[i].module = l->info.module;
            out->pStages = l->stages;
         }
      }
   }
   return l;
}

static void
drop_module(struct wrapper_device *device, VkShaderModule module)
{
   if (!module)
      return;
   wrapper_stride_forget_module(device, module);
   device->dispatch_table.DestroyShaderModule(device->dispatch_handle, module, NULL);
}

void
wrapper_divisor_record(struct wrapper_device *device, void *lowering, VkPipeline pipeline)
{
   struct wrapper_divisor_lowering *l = lowering;
   if (!l)
      return;
   if (pipeline == VK_NULL_HANDLE || !l->info.mask) {
      drop_module(device, l->info.module);
      return;
   }
   struct wrapper_divisor_pipeline *p = malloc(sizeof(*p));
   if (!p) {
      drop_module(device, l->info.module);
      return;
   }
   *p = l->info;
   simple_mtx_lock(&device->resource_mutex);
   _mesa_hash_table_u64_insert(device->divisor_pipeline_table, (uint64_t)pipeline, p);
   simple_mtx_unlock(&device->resource_mutex);
}

void
wrapper_divisor_destroy(struct wrapper_device *device, VkPipeline pipeline)
{
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_divisor_pipeline *p =
      _mesa_hash_table_u64_search(device->divisor_pipeline_table, (uint64_t)pipeline);
   if (p)
      _mesa_hash_table_u64_remove(device->divisor_pipeline_table, (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);
   if (p)
      drop_module(device, p->module);
   free(p);
}

void
wrapper_divisor_bind(struct wrapper_command_buffer *wcb, VkPipeline pipeline)
{
   struct wrapper_device *device = wcb->device;
   simple_mtx_lock(&device->resource_mutex);
   wcb->divisor_pipeline =
      _mesa_hash_table_u64_search(device->divisor_pipeline_table, (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);
}

void
wrapper_divisor_track(struct wrapper_command_buffer *wcb, uint32_t first, uint32_t count,
                      const VkBuffer *buffers, const VkDeviceSize *offsets,
                      const VkDeviceSize *sizes, const VkDeviceSize *strides)
{
   for (uint32_t i = 0; i < count && first + i < WRAPPER_DIVISOR_MAX_BINDINGS; i++) {
      uint32_t b = first + i;
      wcb->vb_buffer[b] = buffers[i];
      wcb->vb_offset[b] = offsets[i];
      wcb->vb_size[b] = sizes ? sizes[i] : VK_WHOLE_SIZE;
      if (strides)
         wcb->vb_stride[b] = strides[i];
      wcb->vb_valid |= 1u << b;
   }
}

void
wrapper_divisor_reset(struct wrapper_command_buffer *wcb)
{
   wcb->divisor_pipeline = NULL;
   wcb->vb_valid = 0;
}

void
wrapper_divisor_finish(struct wrapper_device *device)
{
   hash_table_u64_foreach(device->divisor_pipeline_table, entry)
      free(entry.data);
   _mesa_hash_table_u64_destroy(device->divisor_pipeline_table);
   device->divisor_pipeline_table = NULL;
}

static uint32_t
gcd(uint32_t a, uint32_t b)
{
   while (b) {
      uint32_t t = a % b;
      a = b;
      b = t;
   }
   return a;
}

static void
bind_one(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
         uint32_t b, VkDeviceSize offset, VkDeviceSize size, VkDeviceSize stride)
{
   struct wrapper_device *device = wcb->device;
   VkBuffer buffer = wcb->vb_buffer[b];
   if (p->dynamic_stride && !device->emulate_dynamic_stride &&
       device->dispatch_table.CmdBindVertexBuffers2) {
      device->dispatch_table.CmdBindVertexBuffers2(wcb->dispatch_handle, b, 1, &buffer,
                                                   &offset, &size, &stride);
   } else {
      device->dispatch_table.CmdBindVertexBuffers(wcb->dispatch_handle, b, 1, &buffer,
                                                  &offset);
   }
}

static VkDeviceSize
binding_stride(const struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
               uint32_t b)
{
   return p->dynamic_stride ? wcb->vb_stride[b] : p->stride[b];
}

static void
override_strides(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
                 uint32_t *saved, uint32_t *saved_valid)
{
   *saved_valid = wcb->vertex_strides_valid;
   u_foreach_bit(b, p->mask) {
      saved[b] = wcb->vertex_strides[b];
      wcb->vertex_strides[b] = 0;
      wcb->vertex_strides_valid |= 1u << b;
   }
}

static void
restore_strides(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
                const uint32_t *saved, uint32_t saved_valid)
{
   u_foreach_bit(b, p->mask)
      wcb->vertex_strides[b] = saved[b];
   wcb->vertex_strides_valid = saved_valid;
}

static void
bind_elements(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
              uint32_t first_instance, uint32_t start)
{
   u_foreach_bit(b, p->mask & wcb->vb_valid) {
      uint32_t d = p->divisor[b];
      VkDeviceSize stride = binding_stride(wcb, p, b);
      VkDeviceSize element = d ? first_instance + start / d : first_instance;
      VkDeviceSize skip = element * stride;
      VkDeviceSize size = wcb->vb_size[b];
      if (size != VK_WHOLE_SIZE)
         size = size > skip ? size - skip : 0;
      bind_one(wcb, p, b, wcb->vb_offset[b] + skip, size, 0);
   }
}

static void
restore_bindings(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p)
{
   u_foreach_bit(b, p->mask & wcb->vb_valid)
      bind_one(wcb, p, b, wcb->vb_offset[b], wcb->vb_size[b], binding_stride(wcb, p, b));
}

bool
wrapper_divisor_draw(struct wrapper_command_buffer *wcb, bool indexed, uint32_t count,
                     uint32_t instance_count, uint32_t first, int32_t vertex_offset,
                     uint32_t first_instance)
{
   const struct wrapper_divisor_pipeline *p = wcb->divisor_pipeline;
   struct wrapper_device *device = wcb->device;
   if (!p || !(p->mask & wcb->vb_valid))
      return false;

   uint32_t group = 0;
   u_foreach_bit(b, p->mask)
      if (p->divisor[b])
         group = group ? gcd(group, p->divisor[b]) : p->divisor[b];
   if (!group)
      group = MAX2(instance_count, 1);

   static bool warned_first;
   if (first_instance && p->module && !warned_first) {
      warned_first = true;
      WRAPPER_LOG(error, "Instanced draw with firstInstance %u and a vertex attribute divisor: "
                         "the shader instance index is off by firstInstance", first_instance);
   }

   uint32_t saved[WRAPPER_DIVISOR_MAX_BINDINGS], saved_valid = 0;
   if (wcb->stride_pipeline) {
      override_strides(wcb, p, saved, &saved_valid);
      wrapper_stride_flush(wcb);
   }

   for (uint32_t start = 0; start < instance_count; start += group) {
      uint32_t n = MIN2(group, instance_count - start);
      bind_elements(wcb, p, first_instance, start);
      if (indexed)
         device->dispatch_table.CmdDrawIndexed(wcb->dispatch_handle, count, n, first,
                                               vertex_offset, first_instance + start);
      else
         device->dispatch_table.CmdDraw(wcb->dispatch_handle, count, n, first,
                                        first_instance + start);
   }

   restore_bindings(wcb, p);
   if (wcb->stride_pipeline)
      restore_strides(wcb, p, saved, saved_valid);
   return true;
}

bool
wrapper_divisor_begin_indirect(struct wrapper_command_buffer *wcb)
{
   const struct wrapper_divisor_pipeline *p = wcb->divisor_pipeline;
   if (!p || !(p->mask & wcb->vb_valid))
      return false;
   static bool warned;
   u_foreach_bit(b, p->mask) {
      if (p->divisor[b] > 1 && !warned) {
         warned = true;
         WRAPPER_LOG(error, "Indirect draw with vertex attribute divisor %u: "
                            "every instance reads the first element", p->divisor[b]);
      }
   }
   if (wcb->stride_pipeline) {
      override_strides(wcb, p, wcb->divisor_saved_strides, &wcb->divisor_saved_valid);
      wrapper_stride_flush(wcb);
   }
   bind_elements(wcb, p, 0, 0);
   return true;
}

void
wrapper_divisor_end_indirect(struct wrapper_command_buffer *wcb)
{
   const struct wrapper_divisor_pipeline *p = wcb->divisor_pipeline;
   restore_bindings(wcb, p);
   if (wcb->stride_pipeline)
      restore_strides(wcb, p, wcb->divisor_saved_strides, wcb->divisor_saved_valid);
}
