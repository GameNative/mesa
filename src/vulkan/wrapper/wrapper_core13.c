#include <string.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "vk_common_entrypoints.h"
#include "vk_enum_to_str.h"
#include "vk_extensions.h"
#include "vk_format.h"
#include "vk_physical_device_properties.h"
#include "vk_util.h"
#include "util/u_math.h"
#include "util/u_debug.h"

struct wrapper_promoted_ext {
   const char *name;
   uint32_t version;
   VkStructureType features;
};

static const struct wrapper_promoted_ext wrapper_promoted_exts[] = {
   { NULL, VK_API_VERSION_1_1,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES },
   { NULL, VK_API_VERSION_1_1,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES },
   { NULL, VK_API_VERSION_1_1,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES },
   { NULL, VK_API_VERSION_1_1,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES },
   { NULL, VK_API_VERSION_1_1,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES },
   { "VK_KHR_8bit_storage", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES },
   { "VK_KHR_shader_atomic_int64", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES },
   { "VK_KHR_shader_float16_int8", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES },
   { "VK_EXT_descriptor_indexing", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES },
   { "VK_EXT_scalar_block_layout", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES },
   { "VK_KHR_imageless_framebuffer", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES },
   { "VK_KHR_uniform_buffer_standard_layout", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES },
   { "VK_KHR_shader_subgroup_extended_types", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES },
   { "VK_KHR_separate_depth_stencil_layouts", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES },
   { "VK_EXT_host_query_reset", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES },
   { "VK_KHR_timeline_semaphore", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES },
   { "VK_KHR_buffer_device_address", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES },
   { "VK_KHR_vulkan_memory_model", VK_API_VERSION_1_2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES },
   { "VK_KHR_create_renderpass2", VK_API_VERSION_1_2, 0 },
   { "VK_KHR_depth_stencil_resolve", VK_API_VERSION_1_2, 0 },
   { "VK_KHR_draw_indirect_count", VK_API_VERSION_1_2, 0 },
   { "VK_KHR_sampler_mirror_clamp_to_edge", VK_API_VERSION_1_2, 0 },
   { "VK_EXT_shader_viewport_index_layer", VK_API_VERSION_1_2, 0 },
   { "VK_KHR_driver_properties", VK_API_VERSION_1_2, 0 },
   { "VK_KHR_shader_float_controls", VK_API_VERSION_1_2, 0 },
   { "VK_KHR_spirv_1_4", VK_API_VERSION_1_2, 0 },
   { "VK_EXT_sampler_filter_minmax", VK_API_VERSION_1_2, 0 },
   { "VK_EXT_image_robustness", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES },
   { "VK_EXT_inline_uniform_block", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES },
   { "VK_EXT_pipeline_creation_cache_control", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES },
   { "VK_EXT_private_data", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES },
   { "VK_EXT_shader_demote_to_helper_invocation", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES },
   { "VK_KHR_shader_terminate_invocation", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES },
   { "VK_EXT_subgroup_size_control", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES },
   { "VK_KHR_synchronization2", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES },
   { "VK_KHR_zero_initialize_workgroup_memory", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES },
   { "VK_KHR_dynamic_rendering", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES },
   { "VK_KHR_shader_integer_dot_product", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES },
   { "VK_KHR_maintenance4", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES },
   { "VK_EXT_extended_dynamic_state", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT },
   { "VK_EXT_extended_dynamic_state2", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT },
   { "VK_EXT_texel_buffer_alignment", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT },
   { "VK_EXT_4444_formats", VK_API_VERSION_1_3,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT },
   { "VK_KHR_copy_commands2", VK_API_VERSION_1_3, 0 },
   { "VK_KHR_format_feature_flags2", VK_API_VERSION_1_3, 0 },
   { "VK_KHR_shader_non_semantic_info", VK_API_VERSION_1_3, 0 },
};

static int
wrapper_device_ext_index(const char *name)
{
   for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++)
      if (!strcmp(vk_device_extensions[i].extensionName, name))
         return i;
   return -1;
}

static uint32_t
wrapper_app_api_version(const struct wrapper_physical_device *pdevice)
{
   uint32_t v = pdevice->instance->vk.app_info.api_version;
   return v ? v : VK_API_VERSION_1_0;
}

bool
wrapper_core13_wanted(const struct wrapper_physical_device *pdevice,
                      const char *engine_name)
{
   if (pdevice->properties2.properties.apiVersion >= VK_API_VERSION_1_3)
      return false;
   bool d3d = engine_name &&
      (strstr(engine_name, "DXVK") || strstr(engine_name, "vkd3d"));
   return debug_get_bool_option("WRAPPER_EMULATE_VULKAN_1_3",
      d3d && wrapper_app_api_version(pdevice) >= VK_API_VERSION_1_3);
}

static void
wrapper_query_base_properties(struct wrapper_physical_device *pdevice,
                              VkBaseOutStructure *s)
{
   VkPhysicalDeviceProperties2 p2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = s,
   };
   s->pNext = NULL;
   pdevice->dispatch_table.GetPhysicalDeviceProperties2(pdevice->dispatch_handle, &p2);
   vk_set_physical_device_properties_struct(&pdevice->vk.properties,
                                            (const VkBaseInStructure *)s);
}

#define QUERY_IF(ext, type, stype)                                         \
   if (ext) {                                                              \
      type s = { .sType = stype };                                         \
      wrapper_query_base_properties(pdevice, (VkBaseOutStructure *)&s);    \
   }

void
wrapper_core13_init_physical_device(struct wrapper_physical_device *pdevice)
{
   const struct vk_device_extension_table *e = &pdevice->base_supported_extensions;
   struct vk_features *f = &pdevice->vk.supported_features;
   struct vk_properties *p = &pdevice->vk.properties;

   QUERY_IF(true, VkPhysicalDeviceIDProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
   QUERY_IF(true, VkPhysicalDeviceSubgroupProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
   QUERY_IF(true, VkPhysicalDevicePointClippingProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES);
   QUERY_IF(true, VkPhysicalDeviceMultiviewProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES);
   QUERY_IF(true, VkPhysicalDeviceProtectedMemoryProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES);
   QUERY_IF(true, VkPhysicalDeviceMaintenance3Properties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES);
   QUERY_IF(e->KHR_driver_properties, VkPhysicalDeviceDriverProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
   QUERY_IF(e->KHR_shader_float_controls, VkPhysicalDeviceFloatControlsProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES);
   QUERY_IF(e->EXT_descriptor_indexing, VkPhysicalDeviceDescriptorIndexingProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES);
   QUERY_IF(e->KHR_depth_stencil_resolve, VkPhysicalDeviceDepthStencilResolveProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES);
   QUERY_IF(e->EXT_sampler_filter_minmax, VkPhysicalDeviceSamplerFilterMinmaxProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES);
   QUERY_IF(e->KHR_timeline_semaphore, VkPhysicalDeviceTimelineSemaphoreProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES);
   QUERY_IF(e->EXT_subgroup_size_control, VkPhysicalDeviceSubgroupSizeControlProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
   QUERY_IF(e->EXT_inline_uniform_block, VkPhysicalDeviceInlineUniformBlockProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES);
   QUERY_IF(e->EXT_texel_buffer_alignment, VkPhysicalDeviceTexelBufferAlignmentProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES);
   QUERY_IF(e->KHR_shader_integer_dot_product, VkPhysicalDeviceShaderIntegerDotProductProperties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES);
   QUERY_IF(e->KHR_maintenance4, VkPhysicalDeviceMaintenance4Properties,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES);

   if (!p->maxBufferSize)
      p->maxBufferSize = p->maxMemoryAllocationSize;
   if (!p->minSubgroupSize)
      p->minSubgroupSize = p->subgroupSize;
   if (!p->maxSubgroupSize)
      p->maxSubgroupSize = p->subgroupSize;
   if (!p->maxTimelineSemaphoreValueDifference)
      p->maxTimelineSemaphoreValueDifference = UINT32_MAX;

   f->dynamicRendering = true;
   f->synchronization2 = true;
   f->maintenance4 = true;
   f->shaderZeroInitializeWorkgroupMemory = true;
   f->shaderDemoteToHelperInvocation = true;
   f->descriptorIndexing |= e->EXT_descriptor_indexing;
   f->samplerMirrorClampToEdge |= e->KHR_sampler_mirror_clamp_to_edge;
   f->drawIndirectCount |= e->KHR_draw_indirect_count;
   f->samplerFilterMinmax |= e->EXT_sampler_filter_minmax;
   f->shaderOutputViewportIndex |= e->EXT_shader_viewport_index_layer;
   f->shaderOutputLayer |= e->EXT_shader_viewport_index_layer;
   if (debug_get_bool_option("WRAPPER_EMULATE_EXTENDED_DYNAMIC_STATE", true)) {
      f->extendedDynamicState = true;
      f->extendedDynamicState2 = true;
   }

   uint32_t base = pdevice->properties2.properties.apiVersion;
   WRAPPER_LOG(info, "Lowering Vulkan 1.3 core to a %u.%u.%u base driver",
               VK_API_VERSION_MAJOR(base), VK_API_VERSION_MINOR(base),
               VK_API_VERSION_PATCH(base));
   WRAPPER_LOG(info, "Base update-after-bind limits: per stage samplers %u sampled %u storage img %u "
               "ubo %u ssbo %u resources %u; per set samplers %u sampled %u storage img %u ubo %u ssbo %u; "
               "all pools %u",
               p->maxPerStageDescriptorUpdateAfterBindSamplers,
               p->maxPerStageDescriptorUpdateAfterBindSampledImages,
               p->maxPerStageDescriptorUpdateAfterBindStorageImages,
               p->maxPerStageDescriptorUpdateAfterBindUniformBuffers,
               p->maxPerStageDescriptorUpdateAfterBindStorageBuffers,
               p->maxPerStageUpdateAfterBindResources,
               p->maxDescriptorSetUpdateAfterBindSamplers,
               p->maxDescriptorSetUpdateAfterBindSampledImages,
               p->maxDescriptorSetUpdateAfterBindStorageImages,
               p->maxDescriptorSetUpdateAfterBindUniformBuffers,
               p->maxDescriptorSetUpdateAfterBindStorageBuffers,
               p->maxUpdateAfterBindDescriptorsInAllPools);
   WRAPPER_LOG(info, "Base limits: maxBoundDescriptorSets %u maxPerStageResources %u "
               "maxPushConstantsSize %u", pdevice->properties2.properties.limits.maxBoundDescriptorSets,
               pdevice->properties2.properties.limits.maxPerStageResources,
               pdevice->properties2.properties.limits.maxPushConstantsSize);
}

#undef QUERY_IF

static void
wrapper_fill_one(struct vk_physical_device *source, VkBaseOutStructure *s)
{
   VkPhysicalDeviceProperties2 scratch = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = s,
   };
   void *next = s->pNext;
   s->pNext = NULL;
   vk_common_GetPhysicalDeviceProperties2(vk_physical_device_to_handle(source), &scratch);
   s->pNext = next;
}

void
wrapper_core13_fill_properties(struct wrapper_physical_device *pdevice,
                               VkPhysicalDeviceProperties2 *props)
{
   uint32_t base = pdevice->properties2.properties.apiVersion;
   vk_foreach_struct(s, props->pNext) {
      switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES:
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES:
         if (base < VK_API_VERSION_1_2)
            wrapper_fill_one(&pdevice->vk, s);
         break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES:
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES:
         if (!pdevice->base_supported_extensions.KHR_maintenance4 ||
             s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES)
            wrapper_fill_one(&pdevice->vk, s);
         break;
      default:
         break;
      }
   }
}

void
wrapper_format_properties3_from_legacy(struct wrapper_physical_device *pdevice,
                                       VkFormat format,
                                       VkFormatProperties2 *props)
{
   VkFormatProperties3 *p3 = vk_find_struct(props->pNext, FORMAT_PROPERTIES_3);
   if (!p3)
      return;

   const VkFormatProperties *legacy = &props->formatProperties;
   const struct vk_features *f = &pdevice->base_supported_features;
   VkFormatFeatureFlags2 extra_image = 0, extra_buffer = 0;

   if (f->shaderStorageImageReadWithoutFormat)
      extra_image |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
   if (f->shaderStorageImageWriteWithoutFormat)
      extra_image |= VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
   extra_buffer = extra_image;

   bool depth = vk_format_is_depth_or_stencil(format);

   VkFormatFeatureFlags2 opt = legacy->optimalTilingFeatures;
   VkFormatFeatureFlags2 lin = legacy->linearTilingFeatures;
   VkFormatFeatureFlags2 buf = legacy->bufferFeatures;

   if (opt & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT)
      opt |= extra_image;
   if (lin & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT)
      lin |= extra_image;
   if (buf & VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT)
      buf |= extra_buffer;
   if (depth && (opt & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT))
      opt |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
   if (depth && (lin & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT))
      lin |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;

   p3->optimalTilingFeatures = opt;
   p3->linearTilingFeatures = lin;
   p3->bufferFeatures = buf;
}

static void
wrapper_add_extension(const char **exts, uint32_t *count, const char *name)
{
   for (uint32_t i = 0; i < *count; i++)
      if (!strcmp(exts[i], name))
         return;
   exts[(*count)++] = name;
}

void *
wrapper_core13_lower_device_create(struct wrapper_device *device,
                                   VkDeviceCreateInfo *info,
                                   const char **exts, uint32_t *ext_count)
{
   struct wrapper_physical_device *pdevice = device->physical;
   uint32_t api = wrapper_app_api_version(pdevice);
   struct vk_features req = device->vk.enabled_features;

   if (api >= VK_API_VERSION_1_3) {
      req.extendedDynamicState = true;
      req.extendedDynamicState2 = true;
   }

   bool *r = (bool *)&req;
   const bool *b = (const bool *)&pdevice->base_supported_features;
   for (size_t i = 0; i < sizeof(req) / sizeof(bool); i++)
      r[i] = r[i] && b[i];

   if (pdevice->properties2.properties.apiVersion < VK_API_VERSION_1_2) {
      VkBaseOutStructure *prev = (VkBaseOutStructure *)info;
      while (prev->pNext) {
         if (prev->pNext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
            prev->pNext = prev->pNext->pNext;
         else
            prev = prev->pNext;
      }
   }

   uint32_t n = ARRAY_SIZE(wrapper_promoted_exts);
   void **structs = calloc(n + 1, sizeof(void *));
   if (!structs)
      return NULL;
   VkBaseOutStructure *head = NULL;
   uint32_t used = 0;

   for (uint32_t i = 0; i < n; i++) {
      const struct wrapper_promoted_ext *pe = &wrapper_promoted_exts[i];
      if (pe->version > api)
         continue;
      if (pe->name) {
         int idx = wrapper_device_ext_index(pe->name);
         if (idx < 0 || !pdevice->base_supported_extensions.extensions[idx] ||
             wrapper_filter_extensions.extensions[idx])
            continue;
         wrapper_add_extension(exts, ext_count, vk_device_extensions[idx].extensionName);
      }

      if (!pe->features || __vk_find_struct((void *)info->pNext, pe->features))
         continue;
      VkBaseInStructure probe = { .sType = pe->features };
      VkBaseOutStructure *s = calloc(1, vk_structure_type_size(&probe));
      if (!s)
         continue;
      s->sType = pe->features;
      s->pNext = head;
      head = s;
      structs[used++] = s;
   }

   if (head) {
      struct vk_physical_device *source = malloc(sizeof(*source));
      if (source) {
         *source = pdevice->vk;
         source->supported_features = req;
         VkPhysicalDeviceFeatures2 scratch = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = head,
         };
         vk_common_GetPhysicalDeviceFeatures2(vk_physical_device_to_handle(source),
                                              &scratch);
         free(source);
         VkBaseOutStructure *tail = head;
         while (tail->pNext)
            tail = tail->pNext;
         tail->pNext = (void *)info->pNext;
         info->pNext = head;
      }
   }
   return structs;
}

void
wrapper_core13_free_device_create(void *storage)
{
   void **structs = storage;
   if (!structs)
      return;
   for (uint32_t i = 0; structs[i]; i++)
      free(structs[i]);
   free(structs);
}
