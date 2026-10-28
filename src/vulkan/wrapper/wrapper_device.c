#include <sys/stat.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_bcdec.h"
#include "wrapper_bcn_gpu.h"
#include "spirv_patcher.hpp"
#include "wrapper_spirv_lower.h"
#include "util/ralloc.h"
#include "vk_enum_to_str.h"
#include "wrapper_entrypoints.h"
#include "wrapper_trampolines.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_dispatch_table.h"
#include "vk_extensions.h"
#include "vk_queue.h"
#include "vk_util.h"
#include "util/list.h"
#include "util/simple_mtx.h"

const struct vk_device_extension_table wrapper_device_extensions =
{
   .KHR_swapchain = true,
   .EXT_swapchain_maintenance1 = true,
   .KHR_swapchain_mutable_format = true,
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
   .EXT_display_control = true,
#endif
   .KHR_present_id = true,
   .KHR_present_wait = true,
   .KHR_incremental_present = true,
};

const struct vk_device_extension_table wrapper_filter_extensions =
{
   .EXT_hdr_metadata = true,
   .GOOGLE_display_timing = true,
   .KHR_shared_presentable_image = true,
   .EXT_image_compression_control_swapchain = true,
};

inline struct wrapper_buffer *
get_wrapper_buffer_from_handle_locked(struct wrapper_device *device, VkBuffer buffer) {
   return _mesa_hash_table_u64_search(device->buffer_table, (uint64_t) buffer);
}

struct wrapper_buffer *
get_wrapper_buffer_from_handle(struct wrapper_device *device, VkBuffer buffer) {
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle_locked(device, buffer);
   simple_mtx_unlock(&device->resource_mutex);

   return wb;
}

inline struct wrapper_image *
get_wrapper_image_from_handle_locked(struct wrapper_device *device, VkImage image) {
   return _mesa_hash_table_u64_search(device->image_table, (uint64_t) image);
}

struct wrapper_image *
get_wrapper_image_from_handle(struct wrapper_device *device, VkImage image) {   
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_image *wi = get_wrapper_image_from_handle_locked(device, image);
   simple_mtx_unlock(&device->resource_mutex);
   
   return wi;
}

static struct wrapper_fence *
get_wrapper_fence_from_handle(struct wrapper_device *device, VkFence fence) {
   struct wrapper_fence *wf = NULL;

   simple_mtx_lock(&device->resource_mutex);
   wf = _mesa_hash_table_u64_search(device->fence_table, (uint64_t) fence);
   simple_mtx_unlock(&device->resource_mutex);

   return wf;
}

static void
wrapper_filter_enabled_extensions(const struct wrapper_device *device,
                                  uint32_t *enable_extension_count,
                                  const char **enable_extensions)
{
   for (int idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
      if (!device->vk.enabled_extensions.extensions[idx])
         continue;

      if (!device->physical->base_supported_extensions.extensions[idx])
         continue;

      if (wrapper_device_extensions.extensions[idx])
         continue;

      if (wrapper_filter_extensions.extensions[idx])
         continue;

      enable_extensions[(*enable_extension_count)++] =
         vk_device_extensions[idx].extensionName;
   }

   /* The app enabled one of the vertex_attribute_divisor aliases (both are
    * advertised). Forward whichever one the base driver actually supports;
    * symmetric so a future Mali that gains EXT is handled too. On r44 (neither
    * is present) nothing is forwarded -- the extension is purely spoofed. */
   if (device->vk.enabled_extensions.EXT_vertex_attribute_divisor &&
       !device->vk.enabled_extensions.KHR_vertex_attribute_divisor &&
       device->physical->base_supported_extensions.KHR_vertex_attribute_divisor) {
      enable_extensions[(*enable_extension_count)++] =
         "VK_KHR_vertex_attribute_divisor";
   }

   if (device->vk.enabled_extensions.KHR_vertex_attribute_divisor &&
       !device->vk.enabled_extensions.EXT_vertex_attribute_divisor &&
       device->physical->base_supported_extensions.EXT_vertex_attribute_divisor) {
      enable_extensions[(*enable_extension_count)++] =
         "VK_EXT_vertex_attribute_divisor";
   }
}

static inline void
wrapper_append_required_extensions(const struct wrapper_device *device,
                                  uint32_t *count,
                                  const char **exts) {
#define REQUIRED_EXTENSION(name) \
   if (device->physical->vk.supported_extensions.name) { \
      exts[(*count)++] = "VK_" #name; \
   }
   
   REQUIRED_EXTENSION(KHR_external_fence);
   REQUIRED_EXTENSION(KHR_external_semaphore);
   REQUIRED_EXTENSION(KHR_external_memory);
   REQUIRED_EXTENSION(KHR_external_fence_fd);
   REQUIRED_EXTENSION(KHR_external_semaphore_fd);
   REQUIRED_EXTENSION(KHR_external_memory_fd);
   REQUIRED_EXTENSION(KHR_dedicated_allocation);
   REQUIRED_EXTENSION(EXT_queue_family_foreign);
   REQUIRED_EXTENSION(KHR_maintenance1)
   REQUIRED_EXTENSION(KHR_maintenance2)
   REQUIRED_EXTENSION(KHR_image_format_list)
   REQUIRED_EXTENSION(KHR_swapchain);
   REQUIRED_EXTENSION(KHR_timeline_semaphore);
   /* Vulkan 1.3 promotes these extensions to core.  The wrapper advertises
    * 1.3, but the 1.1 Tegra ICD still requires the EXT names to be enabled
    * before its entry points may be queried. */
   bool lower_core13 = device->physical->emulate_vulkan13;
   if (lower_core13 &&
       !device->vk.enabled_extensions.EXT_extended_dynamic_state &&
       device->physical->base_supported_extensions.EXT_extended_dynamic_state)
      exts[(*count)++] = "VK_EXT_extended_dynamic_state";
   if (lower_core13 &&
       !device->vk.enabled_extensions.EXT_extended_dynamic_state2 &&
       device->physical->base_supported_extensions.EXT_extended_dynamic_state2)
      exts[(*count)++] = "VK_EXT_extended_dynamic_state2";
   REQUIRED_EXTENSION(EXT_external_memory_host);
   REQUIRED_EXTENSION(EXT_external_memory_dma_buf);
   REQUIRED_EXTENSION(EXT_image_drm_format_modifier);
   REQUIRED_EXTENSION(ANDROID_external_memory_android_hardware_buffer);
#undef REQUIRED_EXTENSION
}

static void wrapper_lower_missing_commands(struct wrapper_device *device);

static inline bool
wrapper_uses_variants(const struct wrapper_device *device)
{
   return device->emulate_dynamic_stride || device->emulate_eds || device->emulate_eds2;
}

static inline bool
wrapper_lowers_core13(const struct wrapper_device *device)
{
   return device->physical->emulate_vulkan13;
}

static void unlink_vk_struct(VkBaseInStructure *create_info, const VkBaseInStructure **current, VkBaseInStructure **prev) {
   if (!*prev) 
      create_info->pNext = (*current)->pNext;
   else
      (*prev)->pNext = (*current)->pNext;                                                

   *current = (*current)->pNext;
}

static void process_pnext_chain(VkBaseInStructure *create_info, struct wrapper_physical_device *pdevice) {
   const uint32_t api_version = pdevice->properties2.properties.apiVersion;
   const VkBaseInStructure *current = (VkBaseInStructure *)create_info->pNext;
   VkBaseInStructure *prev = NULL;

   while (current != NULL) {
      switch(current->sType) {
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT: {
             VkPhysicalDeviceTransformFeedbackFeaturesEXT *transform_features =
                (VkPhysicalDeviceTransformFeedbackFeaturesEXT *)current;
             transform_features->geometryStreams &= pdevice->base_supported_features.geometryStreams;
             break;
          }
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_robustness2)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceRobustness2FeaturesEXT from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES:
             if (pdevice->base_supported_features.hostQueryReset)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceHostQueryResetFeatures from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES:
             /* Only when this wrapper advertised the feature itself.  Keying on
              * the driver's extension list instead would strip a legitimate
              * request on a Vulkan 1.2 driver that supports imagelessFramebuffer
              * as core without separately listing VK_KHR_imageless_framebuffer. */
             if (!pdevice->emulate_imageless_framebuffer)
                break;
             WRAPPER_LOG(info,
                "Unlinking emulated VkPhysicalDeviceImagelessFramebufferFeatures from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_dynamic_rendering_unused_attachments)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES:
             if (pdevice->base_supported_extensions.KHR_maintenance5)
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceMaintenance5Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT:
             if (pdevice->base_supported_extensions.EXT_vertex_attribute_divisor)
                break;   /* base has EXT natively -- pass through unchanged */
             if (pdevice->base_supported_extensions.KHR_vertex_attribute_divisor) {
                WRAPPER_LOG(info, "Aliasing VertexAttributeDivisorFeatures EXT->KHR in device pNext");
                ((VkBaseOutStructure *)current)->sType =
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR;
                break;
             }
             /* base supports neither alias (e.g. Mali r44): the extension is
              * purely spoofed, so drop the feature struct rather than hand the
              * real driver a feature for an extension we didn't enable. */
             WRAPPER_LOG(info, "Unlinking VertexAttributeDivisorFeaturesEXT (base lacks it)");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 1, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan11Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 2, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan12Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
             if (api_version >= VK_MAKE_VERSION(1, 3, 0))
                break;
             WRAPPER_LOG(info, "Unlinking VkPhysicalDeviceVulkan13Features from pNext chain");
             unlink_vk_struct(create_info, &current, &prev);
             continue;
          default:
             break;
      }
      prev = (VkBaseInStructure *)current;
      current = current->pNext;
   }
}

/* True when the emulation shares this queue with the application, so submits on
 * it must be serialised against the internal one.  A private queue is invisible
 * to the application, so no wrapper_queue ever carries its handle and this
 * predicate stays false -- the shared-queue cost is paid only in the fallback. */
static bool
wrapper_query_reset_owns_queue(const struct wrapper_queue *queue)
{
   const struct wrapper_device *device = queue->device;
   return device->query_reset_queue != VK_NULL_HANDLE &&
          device->query_reset_shared_queue &&
          device->query_reset_queue == queue->dispatch_handle;
}

/* Reserve a queue for the reset submits.  Preferred: one extra queue in a
 * family the application already asked for, which the application never sees,
 * so a reset cannot wait behind application work.  If the family is exhausted
 * -- the application asked for every queue it has -- fall back to sharing the
 * application's first queue, which is what this emulation did before and is
 * measurably correct but latency-coupled.  Returning an error here instead
 * would turn a slow device into no device at all. */
static VkResult
wrapper_query_reset_reserve_queue(struct wrapper_device *device,
                                  VkDeviceCreateInfo *info,
                                  VkDeviceQueueCreateInfo **queues_out,
                                  float **priorities_out)
{
   struct wrapper_physical_device *physical = device->physical;
   uint32_t count = 0;
   physical->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
      physical->dispatch_handle, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   if (!families)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   physical->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
      physical->dispatch_handle, &count, families);

   /* vkCmdResetQueryPool needs graphics or compute.  flags and pNext are
    * excluded because wrapper_create_device_queue routes those through
    * GetDeviceQueue2, and the shared-queue fallback compares raw handles. */
   int shareable = -1;
   for (uint32_t i = 0; i < info->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *q = &info->pQueueCreateInfos[i];
      uint32_t family = q->queueFamilyIndex;
      if (q->flags || q->pNext || family >= count || !q->queueCount ||
          !(families[family].queueFlags &
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
         continue;
      if (shareable < 0)
         shareable = i;

      uint32_t requested = 0;
      for (uint32_t j = 0; j < info->queueCreateInfoCount; j++)
         if (info->pQueueCreateInfos[j].queueFamilyIndex == family)
            requested += info->pQueueCreateInfos[j].queueCount;
      if (requested >= families[family].queueCount)
         continue;

      VkDeviceQueueCreateInfo *queues =
         malloc(info->queueCreateInfoCount * sizeof(*queues));
      float *priorities = malloc((q->queueCount + 1) * sizeof(*priorities));
      if (!queues || !priorities) {
         free(queues);
         free(priorities);
         free(families);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      memcpy(queues, info->pQueueCreateInfos,
             info->queueCreateInfoCount * sizeof(*queues));
      memcpy(priorities, q->pQueuePriorities, q->queueCount * sizeof(*priorities));
      priorities[q->queueCount] = 0.5f;
      queues[i].queueCount++;
      queues[i].pQueuePriorities = priorities;
      device->query_reset_queue_family = family;
      device->query_reset_queue_index = q->queueCount;
      device->query_reset_shared_queue = false;
      info->pQueueCreateInfos = queues;
      *queues_out = queues;
      *priorities_out = priorities;
      free(families);
      return VK_SUCCESS;
   }

   if (shareable >= 0) {
      device->query_reset_queue_family =
         info->pQueueCreateInfos[shareable].queueFamilyIndex;
      device->query_reset_queue_index = 0;
      device->query_reset_shared_queue = true;
      free(families);
      WRAPPER_LOG(info, "No spare queue for vkResetQueryPool emulation, "
                        "sharing the application queue in family %u",
                  device->query_reset_queue_family);
      return VK_SUCCESS;
   }

   free(families);
   WRAPPER_LOG(error, "vkResetQueryPool emulation requires a graphics or compute queue");
   return VK_ERROR_INITIALIZATION_FAILED;
}

/* A queue of its own for the BCn shaders' self-test, so a shader is checked
 * right after its build on the worker rather than at the application's next
 * submit: one more queue in the first graphics or compute family the
 * application asked for, when the family has one to spare. Otherwise the
 * private queue of the query reset emulation is shared, or the check waits
 * for a submit. */
static void
wrapper_bcn_reserve_queue(struct wrapper_device *device, VkDeviceCreateInfo *info,
                          VkDeviceQueueCreateInfo **queues_out, float **priorities_out)
{
   struct wrapper_physical_device *physical = device->physical;
   uint32_t count = 0;
   physical->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
      physical->dispatch_handle, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   if (!families)
      return;
   physical->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
      physical->dispatch_handle, &count, families);
   for (uint32_t i = 0; i < info->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *q = &info->pQueueCreateInfos[i];
      uint32_t family = q->queueFamilyIndex;
      if (q->flags || q->pNext || family >= count || !q->queueCount ||
          !(families[family].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
         continue;
      uint32_t requested = 0;
      for (uint32_t j = 0; j < info->queueCreateInfoCount; j++)
         if (info->pQueueCreateInfos[j].queueFamilyIndex == family)
            requested += info->pQueueCreateInfos[j].queueCount;
      if (requested >= families[family].queueCount)
         break;
      VkDeviceQueueCreateInfo *queues = malloc(info->queueCreateInfoCount * sizeof(*queues));
      float *priorities = malloc((q->queueCount + 1) * sizeof(*priorities));
      if (!queues || !priorities) {
         free(queues);
         free(priorities);
         break;
      }
      memcpy(queues, info->pQueueCreateInfos, info->queueCreateInfoCount * sizeof(*queues));
      memcpy(priorities, q->pQueuePriorities, q->queueCount * sizeof(*priorities));
      priorities[q->queueCount] = 0.5f;
      queues[i].queueCount++;
      queues[i].pQueuePriorities = priorities;
      device->bcn_queue_family = family;
      device->bcn_queue_index = q->queueCount;
      device->bcn_queue_reserved = true;
      info->pQueueCreateInfos = queues;
      *queues_out = queues;
      *priorities_out = priorities;
      break;
   }
   free(families);
}

static VkResult
wrapper_query_reset_init(struct wrapper_device *device)
{
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   VkDevice dev = device->dispatch_handle;
   dt->GetDeviceQueue(dev, device->query_reset_queue_family,
                      device->query_reset_queue_index, &device->query_reset_queue);

   VkResult result = dt->CreateCommandPool(dev, &(VkCommandPoolCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
               VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = device->query_reset_queue_family,
   }, NULL, &device->query_reset_pool);
   if (result != VK_SUCCESS)
      return result;

   result = dt->AllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = device->query_reset_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   }, &device->query_reset_cmd);
   if (result != VK_SUCCESS)
      return result;

   result = dt->CreateFence(dev, &(VkFenceCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   }, NULL, &device->query_reset_fence);
   if (result == VK_SUCCESS)
      WRAPPER_LOG(info, "Emulating vkResetQueryPool on %s queue (family=%u index=%u)",
                  device->query_reset_shared_queue ? "the application's" : "a private",
                  device->query_reset_queue_family, device->query_reset_queue_index);
   return result;
}

static void
wrapper_query_reset_finish(struct wrapper_device *device)
{
   if (device->query_reset_fence)
      device->dispatch_table.DestroyFence(device->dispatch_handle,
                                          device->query_reset_fence, NULL);
   if (device->query_reset_pool)
      device->dispatch_table.DestroyCommandPool(device->dispatch_handle,
                                                device->query_reset_pool, NULL);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_ResetQueryPool(VkDevice _device, VkQueryPool queryPool,
                       uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   VkResult result;

   if (device->physical->base_supported_features.hostQueryReset) {
      dt->ResetQueryPool(device->dispatch_handle, queryPool, firstQuery, queryCount);
      return;
   }
   /* Not emulating and the driver has no host reset: the application cannot have
    * enabled the feature, because it was never advertised, so this is
    * unreachable through valid usage.  There is no entry point to forward to
    * either -- dispatch_table.ResetQueryPool is NULL on such a driver. */
   if (!queryCount || !device->query_reset_queue)
      return;

   simple_mtx_lock(&device->query_reset_mutex);
   if (vk_device_is_lost(&device->vk)) {
      simple_mtx_unlock(&device->query_reset_mutex);
      return;
   }
   result = dt->ResetCommandBuffer(device->query_reset_cmd, 0);
   if (result != VK_SUCCESS)
      goto fail;
   result = dt->BeginCommandBuffer(device->query_reset_cmd,
      &(VkCommandBufferBeginInfo) {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      });
   if (result != VK_SUCCESS)
      goto fail;
   dt->CmdResetQueryPool(device->query_reset_cmd, queryPool, firstQuery, queryCount);
   result = dt->EndCommandBuffer(device->query_reset_cmd);
   if (result != VK_SUCCESS)
      goto fail;
   result = dt->ResetFences(device->dispatch_handle, 1, &device->query_reset_fence);
   if (result != VK_SUCCESS)
      goto fail;
   result = dt->QueueSubmit(device->query_reset_queue, 1, &(VkSubmitInfo) {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &device->query_reset_cmd,
   }, device->query_reset_fence);
   if (result != VK_SUCCESS)
      goto fail;
   result = dt->WaitForFences(device->dispatch_handle, 1,
                              &device->query_reset_fence, VK_TRUE, UINT64_MAX);
   if (result != VK_SUCCESS)
      goto fail;
   simple_mtx_unlock(&device->query_reset_mutex);
   return;

fail:
   vk_device_set_lost(&device->vk, "Host query reset emulation failed: %d", result);
   simple_mtx_unlock(&device->query_reset_mutex);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_DeviceWaitIdle(VkDevice _device)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (!device->query_reset_queue)
      return device->dispatch_table.DeviceWaitIdle(device->dispatch_handle);

   simple_mtx_lock(&device->query_reset_mutex);
   VkResult result = vk_device_is_lost(&device->vk) ? VK_ERROR_DEVICE_LOST :
      device->dispatch_table.DeviceWaitIdle(device->dispatch_handle);
   simple_mtx_unlock(&device->query_reset_mutex);
   return result;
}

static VkResult
wrapper_create_device_queue(struct wrapper_device *device,
                            const VkDeviceCreateInfo* pCreateInfo)
{
   const VkDeviceQueueCreateInfo *create_info;
   struct wrapper_queue *queue;
   VkResult result;

   for (int i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      create_info = &pCreateInfo->pQueueCreateInfos[i];
      for (int j = 0; j < create_info->queueCount; j++) {
         queue = vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                           VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
         if (!queue)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

         if (create_info->flags) {
            device->dispatch_table.GetDeviceQueue2(
               device->dispatch_handle,
               &(VkDeviceQueueInfo2) {
                  .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
                  .flags = create_info->flags,
                  .queueFamilyIndex = create_info->queueFamilyIndex,
                  .queueIndex = j,
               },
               &queue->dispatch_handle);;
         } else {
            device->dispatch_table.GetDeviceQueue(
               device->dispatch_handle, create_info->queueFamilyIndex,
               j, &queue->dispatch_handle);
         }
         queue->device = device;

         result = vk_queue_init(&queue->vk, &device->vk, create_info, j);
         if (result != VK_SUCCESS) {
            vk_free(&device->vk.alloc, queue);
            return result;
         }
      }
   }

   return VK_SUCCESS;
}

static void
wrapper_create_null_resources(struct wrapper_device *device)
{
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   VkDevice dev = device->dispatch_handle;
   VkPhysicalDeviceMemoryProperties *mp = &device->physical->memory_properties;

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 65536,
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (dt->CreateBuffer(dev, &bci, NULL, &device->null_buffer) != VK_SUCCESS)
      return;
   VkMemoryRequirements mr;
   dt->GetBufferMemoryRequirements(dev, device->null_buffer, &mr);
   uint32_t mt = UINT32_MAX;
   for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { mt = i; break; }
   if (mt == UINT32_MAX) return;
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt,
   };
   if (dt->AllocateMemory(dev, &mai, NULL, &device->null_buffer_memory) != VK_SUCCESS)
      return;
   dt->BindBufferMemory(dev, device->null_buffer, device->null_buffer_memory, 0);
   void *ptr = NULL;
   if (dt->MapMemory(dev, device->null_buffer_memory, 0, VK_WHOLE_SIZE, 0, &ptr) == VK_SUCCESS) {
      memset(ptr, 0, mr.size);
      dt->UnmapMemory(dev, device->null_buffer_memory);
   }

   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { 1, 1, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   if (dt->CreateImage(dev, &ici, NULL, &device->null_image) == VK_SUCCESS) {
      VkMemoryRequirements imr;
      dt->GetImageMemoryRequirements(dev, device->null_image, &imr);
      uint32_t imt = UINT32_MAX;
      for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
         if (imr.memoryTypeBits & (1u << i)) { imt = i; break; }
      VkMemoryAllocateInfo imai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = imr.size, .memoryTypeIndex = imt,
      };
      if (imt != UINT32_MAX &&
          dt->AllocateMemory(dev, &imai, NULL, &device->null_image_memory) == VK_SUCCESS) {
         dt->BindImageMemory(dev, device->null_image, device->null_image_memory, 0);
         VkImageViewCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = device->null_image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
         };
         dt->CreateImageView(dev, &vci, NULL, &device->null_image_view);
      }
   }
   VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
   dt->CreateSampler(dev, &sci, NULL, &device->null_sampler);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_UpdateDescriptorSets(VkDevice _device, uint32_t descriptorWriteCount,
                             const VkWriteDescriptorSet *pDescriptorWrites,
                             uint32_t descriptorCopyCount,
                             const VkCopyDescriptorSet *pDescriptorCopies)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkWriteDescriptorSet *fold_writes = (VkWriteDescriptorSet *)pDescriptorWrites;
   VkCopyDescriptorSet *fold_copies = (VkCopyDescriptorSet *)pDescriptorCopies;
   if (device->fold_sets &&
       wrapper_fold_remap_updates(device, &descriptorWriteCount, &fold_writes,
                                  &descriptorCopyCount, &fold_copies)) {
      wrapper_UpdateDescriptorSets(_device, descriptorWriteCount, fold_writes,
                                   descriptorCopyCount, fold_copies);
      free(fold_writes);
      free(fold_copies);
      return;
   }

   if (!device->emulate_null_descriptor || descriptorWriteCount == 0) {
      device->dispatch_table.UpdateDescriptorSets(device->dispatch_handle,
         descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
      return;
   }

   VkWriteDescriptorSet *writes =
      malloc(sizeof(VkWriteDescriptorSet) * descriptorWriteCount);
   memcpy(writes, pDescriptorWrites, sizeof(VkWriteDescriptorSet) * descriptorWriteCount);

   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *w = &pDescriptorWrites[i];
      uint32_t n = w->descriptorCount;
      switch (w->descriptorType) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
         VkDescriptorBufferInfo *bi = malloc(sizeof(*bi) * n);
         memcpy(bi, w->pBufferInfo, sizeof(*bi) * n);
         for (uint32_t j = 0; j < n; j++)
            if (bi[j].buffer == VK_NULL_HANDLE) {
               bi[j].buffer = device->null_buffer;
               bi[j].offset = 0; bi[j].range = VK_WHOLE_SIZE;
            }
         writes[i].pBufferInfo = bi;
         break;
      }
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      case VK_DESCRIPTOR_TYPE_SAMPLER: {
         VkDescriptorImageInfo *ii = malloc(sizeof(*ii) * n);
         memcpy(ii, w->pImageInfo, sizeof(*ii) * n);
         for (uint32_t j = 0; j < n; j++) {
            if ((w->descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) &&
                ii[j].imageView == VK_NULL_HANDLE) {
               ii[j].imageView = device->null_image_view;
               ii[j].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            if ((w->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                 w->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
                ii[j].sampler == VK_NULL_HANDLE)
               ii[j].sampler = device->null_sampler;
         }
         writes[i].pImageInfo = ii;
         break;
      }
      default:
         break;
      }
   }

   device->dispatch_table.UpdateDescriptorSets(device->dispatch_handle,
      descriptorWriteCount, writes, descriptorCopyCount, pDescriptorCopies);

   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      if (writes[i].pBufferInfo != pDescriptorWrites[i].pBufferInfo)
         free((void *)writes[i].pBufferInfo);
      if (writes[i].pImageInfo != pDescriptorWrites[i].pImageInfo)
         free((void *)writes[i].pImageInfo);
   }
   free(writes);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                             uint32_t bindingCount, const VkBuffer *pBuffers,
                             const VkDeviceSize *pOffsets, const VkDeviceSize *pSizes,
                             const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkBuffer *bufs = NULL;
   VkDeviceSize *offsets = NULL, *sizes = NULL;
   if (device->emulate_null_descriptor && pBuffers) {
      bufs = malloc(sizeof(VkBuffer) * bindingCount);
      offsets = malloc(sizeof(VkDeviceSize) * bindingCount);
      sizes = pSizes ? malloc(sizeof(VkDeviceSize) * bindingCount) : NULL;
      for (uint32_t i = 0; i < bindingCount; i++) {
         bufs[i] = pBuffers[i] ? pBuffers[i] : device->null_buffer;
         offsets[i] = pBuffers[i] ? pOffsets[i] : 0;
         if (sizes)
            sizes[i] = pBuffers[i] ? pSizes[i] : VK_WHOLE_SIZE;
      }
      pBuffers = bufs;
      pOffsets = offsets;
      if (sizes)
         pSizes = sizes;
   }
   if ((device->emulate_vertex_divisor || device->emulate_zero_divisor) && pBuffers)
      wrapper_divisor_track(wcb, firstBinding, bindingCount, pBuffers, pOffsets, pSizes,
                            pStrides);
   if (device->emulate_dynamic_stride && pStrides) {
      wrapper_stride_set(wcb, firstBinding, bindingCount, pStrides);
      pStrides = NULL;
   }
   if (device->dispatch_table.CmdBindVertexBuffers2)
      device->dispatch_table.CmdBindVertexBuffers2(wcb->dispatch_handle,
         firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides);
   else if (wrapper_lowers_core13(device) &&
            device->dispatch_table.CmdBindVertexBuffers2EXT)
      device->dispatch_table.CmdBindVertexBuffers2EXT(wcb->dispatch_handle,
         firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides);
   else if (bindingCount)
      device->dispatch_table.CmdBindVertexBuffers(wcb->dispatch_handle,
         firstBinding, bindingCount, pBuffers, pOffsets);
   free(bufs);
   free(offsets);
   free(sizes);
}

/* Core 1.3 dynamic state may be supplied by EXT_extended_dynamic_state{,2} on
 * an older base driver. Define only the core names: Mesa aliases core and EXT in
 * its public dispatch table, so defining both would register the same slot
 * twice. */
VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetCullMode(VkCommandBuffer commandBuffer, VkCullModeFlags cullMode)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetCullMode)
      wcb->device->dispatch_table.CmdSetCullMode(wcb->dispatch_handle, cullMode);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetCullModeEXT)
      wcb->device->dispatch_table.CmdSetCullModeEXT(wcb->dispatch_handle, cullMode);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_CULL, cullMode);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetFrontFace(VkCommandBuffer commandBuffer, VkFrontFace frontFace)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetFrontFace)
      wcb->device->dispatch_table.CmdSetFrontFace(wcb->dispatch_handle, frontFace);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetFrontFaceEXT)
      wcb->device->dispatch_table.CmdSetFrontFaceEXT(wcb->dispatch_handle, frontFace);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_FRONT, frontFace);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetPrimitiveTopology(VkCommandBuffer commandBuffer,
                                VkPrimitiveTopology primitiveTopology)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetPrimitiveTopology)
      wcb->device->dispatch_table.CmdSetPrimitiveTopology(
         wcb->dispatch_handle, primitiveTopology);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetPrimitiveTopologyEXT)
      wcb->device->dispatch_table.CmdSetPrimitiveTopologyEXT(
         wcb->dispatch_handle, primitiveTopology);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_TOPOLOGY, primitiveTopology);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetViewportWithCount(VkCommandBuffer commandBuffer,
                                uint32_t viewportCount,
                                const VkViewport *pViewports)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetViewportWithCount)
      wcb->device->dispatch_table.CmdSetViewportWithCount(
         wcb->dispatch_handle, viewportCount, pViewports);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetViewportWithCountEXT)
      wcb->device->dispatch_table.CmdSetViewportWithCountEXT(
         wcb->dispatch_handle, viewportCount, pViewports);
   else if (wcb->device->emulate_eds && viewportCount) {
      uint32_t count = MIN2(viewportCount, wcb->device->max_viewports);
      wcb->device->dispatch_table.CmdSetViewport(wcb->dispatch_handle, 0, count, pViewports);
      wrapper_eds_set(wcb, WRAPPER_EDS_VIEWPORTS, count);
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetScissorWithCount(VkCommandBuffer commandBuffer,
                               uint32_t scissorCount,
                               const VkRect2D *pScissors)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetScissorWithCount)
      wcb->device->dispatch_table.CmdSetScissorWithCount(
         wcb->dispatch_handle, scissorCount, pScissors);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetScissorWithCountEXT)
      wcb->device->dispatch_table.CmdSetScissorWithCountEXT(
         wcb->dispatch_handle, scissorCount, pScissors);
   else if (wcb->device->emulate_eds && scissorCount) {
      uint32_t count = MIN2(scissorCount, wcb->device->max_viewports);
      wcb->device->dispatch_table.CmdSetScissor(wcb->dispatch_handle, 0, count, pScissors);
      wrapper_eds_set(wcb, WRAPPER_EDS_SCISSORS, count);
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetDepthTestEnable(VkCommandBuffer commandBuffer,
                              VkBool32 depthTestEnable)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetDepthTestEnable)
      wcb->device->dispatch_table.CmdSetDepthTestEnable(
         wcb->dispatch_handle, depthTestEnable);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetDepthTestEnableEXT)
      wcb->device->dispatch_table.CmdSetDepthTestEnableEXT(
         wcb->dispatch_handle, depthTestEnable);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_DEPTH_TEST, depthTestEnable);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetDepthWriteEnable(VkCommandBuffer commandBuffer,
                               VkBool32 depthWriteEnable)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetDepthWriteEnable)
      wcb->device->dispatch_table.CmdSetDepthWriteEnable(
         wcb->dispatch_handle, depthWriteEnable);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetDepthWriteEnableEXT)
      wcb->device->dispatch_table.CmdSetDepthWriteEnableEXT(
         wcb->dispatch_handle, depthWriteEnable);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_DEPTH_WRITE, depthWriteEnable);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetDepthCompareOp(VkCommandBuffer commandBuffer,
                             VkCompareOp depthCompareOp)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetDepthCompareOp)
      wcb->device->dispatch_table.CmdSetDepthCompareOp(
         wcb->dispatch_handle, depthCompareOp);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetDepthCompareOpEXT)
      wcb->device->dispatch_table.CmdSetDepthCompareOpEXT(
         wcb->dispatch_handle, depthCompareOp);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_DEPTH_COMPARE, depthCompareOp);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetStencilTestEnable(VkCommandBuffer commandBuffer,
                                VkBool32 stencilTestEnable)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetStencilTestEnable)
      wcb->device->dispatch_table.CmdSetStencilTestEnable(
         wcb->dispatch_handle, stencilTestEnable);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetStencilTestEnableEXT)
      wcb->device->dispatch_table.CmdSetStencilTestEnableEXT(
         wcb->dispatch_handle, stencilTestEnable);
   else if (wcb->device->emulate_eds)
      wrapper_eds_set(wcb, WRAPPER_EDS_STENCIL_TEST, stencilTestEnable);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetStencilOp(VkCommandBuffer commandBuffer,
                        VkStencilFaceFlags faceMask, VkStencilOp failOp,
                        VkStencilOp passOp, VkStencilOp depthFailOp,
                        VkCompareOp compareOp)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetStencilOp)
      wcb->device->dispatch_table.CmdSetStencilOp(
         wcb->dispatch_handle, faceMask, failOp, passOp, depthFailOp, compareOp);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetStencilOpEXT)
      wcb->device->dispatch_table.CmdSetStencilOpEXT(
         wcb->dispatch_handle, faceMask, failOp, passOp, depthFailOp, compareOp);
   else if (wcb->device->emulate_eds) {
      if (faceMask & VK_STENCIL_FACE_FRONT_BIT) {
         wrapper_eds_set(wcb, WRAPPER_EDS_FRONT_FAIL, failOp);
         wrapper_eds_set(wcb, WRAPPER_EDS_FRONT_PASS, passOp);
         wrapper_eds_set(wcb, WRAPPER_EDS_FRONT_DEPTH_FAIL, depthFailOp);
         wrapper_eds_set(wcb, WRAPPER_EDS_FRONT_COMPARE, compareOp);
      }
      if (faceMask & VK_STENCIL_FACE_BACK_BIT) {
         wrapper_eds_set(wcb, WRAPPER_EDS_BACK_FAIL, failOp);
         wrapper_eds_set(wcb, WRAPPER_EDS_BACK_PASS, passOp);
         wrapper_eds_set(wcb, WRAPPER_EDS_BACK_DEPTH_FAIL, depthFailOp);
         wrapper_eds_set(wcb, WRAPPER_EDS_BACK_COMPARE, compareOp);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetRasterizerDiscardEnable(VkCommandBuffer commandBuffer,
                                      VkBool32 rasterizerDiscardEnable)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetRasterizerDiscardEnable)
      wcb->device->dispatch_table.CmdSetRasterizerDiscardEnable(
         wcb->dispatch_handle, rasterizerDiscardEnable);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetRasterizerDiscardEnableEXT)
      wcb->device->dispatch_table.CmdSetRasterizerDiscardEnableEXT(
         wcb->dispatch_handle, rasterizerDiscardEnable);
   else if (wcb->device->emulate_eds2)
      wrapper_eds_set(wcb, WRAPPER_EDS_RASTER_DISCARD, rasterizerDiscardEnable);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetDepthBiasEnable(VkCommandBuffer commandBuffer,
                              VkBool32 depthBiasEnable)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetDepthBiasEnable)
      wcb->device->dispatch_table.CmdSetDepthBiasEnable(
         wcb->dispatch_handle, depthBiasEnable);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetDepthBiasEnableEXT)
      wcb->device->dispatch_table.CmdSetDepthBiasEnableEXT(
         wcb->dispatch_handle, depthBiasEnable);
   else if (wcb->device->emulate_eds2)
      wrapper_eds_set(wcb, WRAPPER_EDS_DEPTH_BIAS, depthBiasEnable);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdSetPrimitiveRestartEnable(VkCommandBuffer commandBuffer,
                                     VkBool32 primitiveRestartEnable)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdSetPrimitiveRestartEnable)
      wcb->device->dispatch_table.CmdSetPrimitiveRestartEnable(
         wcb->dispatch_handle, primitiveRestartEnable);
   else if (wrapper_lowers_core13(wcb->device) &&
            wcb->device->dispatch_table.CmdSetPrimitiveRestartEnableEXT)
      wcb->device->dispatch_table.CmdSetPrimitiveRestartEnableEXT(
         wcb->dispatch_handle, primitiveRestartEnable);
   else if (wcb->device->emulate_eds2)
      wrapper_eds_set(wcb, WRAPPER_EDS_PRIM_RESTART, primitiveRestartEnable);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                            uint32_t bindingCount, const VkBuffer *pBuffers,
                            const VkDeviceSize *pOffsets)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkBuffer *bufs = NULL;
   VkDeviceSize *offsets = NULL;
   if (device->emulate_null_descriptor && pBuffers) {
      bufs = malloc(sizeof(VkBuffer) * bindingCount);
      offsets = malloc(sizeof(VkDeviceSize) * bindingCount);
      for (uint32_t i = 0; i < bindingCount; i++) {
         bufs[i] = pBuffers[i] ? pBuffers[i] : device->null_buffer;
         offsets[i] = pBuffers[i] ? pOffsets[i] : 0;
      }
   }
   if ((device->emulate_vertex_divisor || device->emulate_zero_divisor) && pBuffers)
      wrapper_divisor_track(wcb, firstBinding, bindingCount, bufs ? bufs : pBuffers,
                            offsets ? offsets : pOffsets, NULL, NULL);
   device->dispatch_table.CmdBindVertexBuffers(wcb->dispatch_handle, firstBinding,
      bindingCount, bufs ? bufs : pBuffers, offsets ? offsets : pOffsets);
   free(bufs);
   free(offsets);
}


/* Images with dropped top mips (policy maxdim); zero keeps every
 * remap hook off the lookup path. */
static int wrapper_capped_images;

static uint32_t
wrapper_image_mip_drop(struct wrapper_device *device, VkImage image)
{
   if (!__atomic_load_n(&wrapper_capped_images, __ATOMIC_RELAXED))
      return 0;
   struct wrapper_image *wi = get_wrapper_image_from_handle(device, image);
   return wi ? wi->mip_drop : 0;
}

#define WRAPPER_CAP_STACK_ENTRIES 16

static void *
wrapper_cap_alloc(void *stack, size_t size, uint32_t count)
{
   return count <= WRAPPER_CAP_STACK_ENTRIES ? stack : malloc(size * count);
}

static void
wrapper_cap_free(void *array, void *stack)
{
   if (array != stack)
      free(array);
}

static bool
wrapper_cap_level(uint32_t mip_drop, uint32_t *level)
{
   if (*level < mip_drop)
      return false;
   *level -= mip_drop;
   return true;
}

/* Barriers on dropped levels are removed; returns the kept count. */
static uint32_t
wrapper_cap_image_barriers(struct wrapper_device *device,
                           const VkImageMemoryBarrier *in, uint32_t count,
                           VkImageMemoryBarrier *out)
{
   uint32_t n = 0;
   for (uint32_t i = 0; i < count; i++) {
      out[n] = in[i];
      if (bcn_cap_range(wrapper_image_mip_drop(device, in[i].image),
                        &out[n].subresourceRange))
         n++;
   }
   return n;
}

static uint32_t
wrapper_cap_image_barriers2(struct wrapper_device *device,
                            const VkImageMemoryBarrier2 *in, uint32_t count,
                            VkImageMemoryBarrier2 *out)
{
   uint32_t n = 0;
   for (uint32_t i = 0; i < count; i++) {
      out[n] = in[i];
      if (bcn_cap_range(wrapper_image_mip_drop(device, in[i].image),
                        &out[n].subresourceRange))
         n++;
   }
   return n;
}

static const char *
wrapper_driver_id_str(VkDriverId id)
{
   switch (id) {
   case VK_DRIVER_ID_ARM_PROPRIETARY:         return "ARM (Mali)";
   case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:    return "Qualcomm (Adreno)";
   case VK_DRIVER_ID_SAMSUNG_PROPRIETARY:     return "Samsung (Xclipse)";
   case VK_DRIVER_ID_MESA_TURNIP:             return "Mesa Turnip (Adreno)";
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY: return "Imagination (PowerVR)";
   default:                                   return "other/unknown";
   }
}

/* WRAPPER_DIAG=1 emits a self-contained device-capabilities report — to
 * stderr (so it lands in logcat) and to a file (WRAPPER_DIAG_FILE, or
 * $TMPDIR/wrapper_diag.txt) a user can share. It reports what the device
 * NATIVELY supports vs what the wrapper advertises to the D3D layer, so a
 * failing game on any device can be diagnosed from the report alone. Pair it
 * with app-side VKD3D_DEBUG=warn / DXVK_LOG_LEVEL=info / WINEDEBUG=+vulkan. */
static int wrapper_diag_stderr = -1;
static bool wrapper_diag_capture;

static void
wrapper_diag_autostart(struct wrapper_physical_device *pdev)
{
   const char *app = pdev->instance->vk.app_info.app_name;
   if (!app || strcmp(app, "Wheel World.exe") ||
       pdev->driver_properties.driverID != VK_DRIVER_ID_IMAGINATION_PROPRIETARY)
      return;

   /* This diagnostic package must work without a launcher environment change.
    * Keep stderr connected to the launcher: component runs may not request the
    * separate wrapper diagnostic attachment. No rendering options change. */
   wrapper_diag_capture = true;
   setenv("WRAPPER_DIAG", "1", 1);
   fprintf(stderr, "[WRAPPER_DIAG] wheel-world-wrapper-diag-v1: "
                   "device creation entered; driver=%s\n",
           pdev->properties2.properties.deviceName);
   fflush(stderr);
}

static void
wrapper_emit_diag(struct wrapper_physical_device *pdev,
                  const VkDeviceCreateInfo *ci, VkResult result)
{
   const char *en = getenv("WRAPPER_DIAG");
   if (!en || !atoi(en))
      return;

   /* Report file lives in the container tmp dir, one per game (unique by app
    * name) so multiple games don't clobber each other. WRAPPER_DIAG_FILE
    * overrides the full path. */
   char tag[128], defpath[512];
   /* Prefer the app id passed by the launcher (WRAPPER_DIAG_APPID); fall back
    * to the executable name so the file is still unique per game either way. */
   const char *app = getenv("WRAPPER_DIAG_APPID");
   if (!app || !app[0]) {
      app = pdev->instance->vk.app_info.app_name;
      if (!app || !app[0])
         app = "unknown";
      const char *bslash = strrchr(app, '\\');   /* strip Windows/Unix path */
      if (bslash) app = bslash + 1;
      const char *fslash = strrchr(app, '/');
      if (fslash) app = fslash + 1;
   }
   size_t j = 0;
   for (size_t i = 0; app[i] && j < sizeof(tag) - 1; i++) {
      char c = app[i];
      int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
               (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
      tag[j++] = ok ? c : '_';
   }
   tag[j] = 0;
   snprintf(defpath, sizeof(defpath),
            "/data/data/app.gamenative/files/imagefs/usr/tmp/wrapper_diag_%s.txt", tag);
   const char *path = getenv("WRAPPER_DIAG_FILE") ? getenv("WRAPPER_DIAG_FILE") : defpath;

   /* Put EVERYTHING needed to diagnose a failing game in ONE shareable file. For
    * a D3D engine, redirect this process's stdout+stderr into the diag file once,
    * so the vkd3d/DXVK/wine output that actually explains the failure
    * (Heap-too-small, device-lost, format rejects, crashes) is captured in the
    * same file as the wrapper's capability report below — instead of only the
    * wrapper's half. Non-D3D processes (zink infra) stay on stderr/logcat, so the
    * file is just the D3D process. usr/tmp is recreated per launch. */
   {
      const char *e = pdev->instance->vk.app_info.engine_name;
      static int redirected = 0;
      if (!wrapper_diag_capture && !redirected && e &&
          (strstr(e, "DXVK") || strstr(e, "vkd3d"))) {
         redirected = 1;
         int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
         if (fd >= 0) {
            wrapper_diag_stderr = dup(2);
            dup2(fd, 1);   /* stdout: vkd3d/wine 'System.out' errors */
            dup2(fd, 2);   /* stderr */
            if (fd > 2) close(fd);
            setvbuf(stdout, NULL, _IONBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
         }
      }
   }

#define D(...) do { \
   if (wrapper_diag_capture) \
      wrapper_diag_append("[WRAPPER_DIAG] " __VA_ARGS__); \
   else \
      fprintf(stderr, "[WRAPPER_DIAG] " __VA_ARGS__); \
} while (0)

   const VkPhysicalDeviceProperties *p = &pdev->properties2.properties;
   const char *eng = pdev->instance->vk.app_info.engine_name
      ? pdev->instance->vk.app_info.engine_name : "";
   uint32_t ev = pdev->instance->vk.app_info.engine_version;
   char detected[64];
   if (strstr(eng, "DXVK"))
      snprintf(detected, sizeof(detected), "DXVK %u.%u.%u",
               VK_VERSION_MAJOR(ev), VK_VERSION_MINOR(ev), VK_VERSION_PATCH(ev));
   else if (strstr(eng, "vkd3d"))
      snprintf(detected, sizeof(detected), "vkd3d %u.%u.%u",
               VK_VERSION_MAJOR(ev), VK_VERSION_MINOR(ev), VK_VERSION_PATCH(ev));
   else
      snprintf(detected, sizeof(detected), "native/other");

   D("\n================ WRAPPER DIAGNOSTICS ================\n");
   D("build: %s %s\n", __DATE__, __TIME__);
   D("device: %s\n", p->deviceName);
   D("  driver=%s (driverID=%u)  driverVersion=0x%08x  apiVersion=%u.%u.%u\n",
     wrapper_driver_id_str(pdev->driver_properties.driverID),
     pdev->driver_properties.driverID, p->driverVersion,
     VK_VERSION_MAJOR(p->apiVersion), VK_VERSION_MINOR(p->apiVersion),
     VK_VERSION_PATCH(p->apiVersion));
   D("  vendorID=0x%04x deviceID=0x%04x\n", p->vendorID, p->deviceID);
   D("engine: '%s' -> detected %s (app '%s')\n", eng, detected,
     pdev->instance->vk.app_info.app_name ? pdev->instance->vk.app_info.app_name : "");
   D("--- native capabilities (what THIS device actually has) ---\n");
   D("  textureCompressionBC       : %d\n", pdev->base_supported_features.textureCompressionBC);
   D("  textureCompressionASTC_LDR : %d\n", pdev->base_supported_features.textureCompressionASTC_LDR);
   D("  robustBufferAccess2        : %d\n", pdev->base_supported_features.robustBufferAccess2);
   D("  nullDescriptor             : %d\n", pdev->base_supported_features.nullDescriptor);
   D("  robustImageAccess2         : %d\n", pdev->base_supported_features.robustImageAccess2);
   D("  extendedDynamicState       : %d\n", pdev->base_supported_features.extendedDynamicState);
   D("  extendedDynamicState2      : %d\n", pdev->base_supported_features.extendedDynamicState2);
   D("  dualSrcBlend               : %d\n", pdev->base_supported_features.dualSrcBlend);
   D("  multiDrawIndirect          : %d\n", pdev->base_supported_features.multiDrawIndirect);
   D("  ext EXT_robustness2              : %d\n", pdev->base_supported_extensions.EXT_robustness2);
   D("  ext EXT_vertex_attribute_divisor : %d\n", pdev->base_supported_extensions.EXT_vertex_attribute_divisor);
   D("  ext KHR_vertex_attribute_divisor : %d\n", pdev->base_supported_extensions.KHR_vertex_attribute_divisor);
   D("--- wrapper overrides advertised to the D3D layer ---\n");
   D("  WRAPPER_VK_VERSION advertised : %s\n",
     getenv("WRAPPER_VK_VERSION") ? getenv("WRAPPER_VK_VERSION") : "(driver default)");
   D("  faking VK_EXT_robustness2     : %s\n",
     (pdev->vk.supported_extensions.EXT_robustness2 && !pdev->base_supported_extensions.EXT_robustness2) ? "YES" : "no");
   D("  vertex_attr_divisor EXT alias : %s\n",
     (pdev->vk.supported_extensions.EXT_vertex_attribute_divisor && !pdev->base_supported_extensions.EXT_vertex_attribute_divisor) ? "YES (aliased from KHR)" : "no");
   D("  BCn: emulate=%d  ASTC=%s  BC1=%s  transcode=%s  cache=%s  upload=%s\n",
     pdev->emulate_bcn,
     getenv("WRAPPER_ASTC_BLOCK") ? getenv("WRAPPER_ASTC_BLOCK") : "4x4",
     is_astc_6x6(get_format_for_bcn(VK_FORMAT_BC1_RGB_UNORM_BLOCK)) ? "6x6" :
     is_astc_8x8(get_format_for_bcn(VK_FORMAT_BC1_RGB_UNORM_BLOCK)) ? "8x8" :
     is_astc_4x4(get_format_for_bcn(VK_FORMAT_BC1_RGB_UNORM_BLOCK)) ? "4x4" : "decode",
     wrapper_bcn_gpu_path() ? "GPU" : "CPU",
     (!getenv("WRAPPER_USE_BCN_CACHE") || atoi(getenv("WRAPPER_USE_BCN_CACHE"))) ? "on" : "off",
     bcn_upload_enabled() ? "on" : "off");
   D("  format diag                   : wrapper-fmtdiag-1\n");
   D("  BCn policy (WRAPPER_BCN_POLICY): %s  BC6H=%s\n", bcn_policy_desc(),
     is_astc_hdr_4x4(get_format_for_bcn(VK_FORMAT_BC6H_UFLOAT_BLOCK)) ? "ASTC 4x4 HDR" : "decoded");
   D("  VK_EXT_device_fault report    : %s\n",
     !pdev->base_supported_extensions.EXT_device_fault ? "unsupported by base driver" :
     (!getenv("WRAPPER_DEVICE_FAULT") || atoi(getenv("WRAPPER_DEVICE_FAULT")))
        ? "ON (GPU fault dumped on device loss)" : "off (WRAPPER_DEVICE_FAULT=0)");
   D("  push_descriptor emulation     : %s\n",
     (getenv("WRAPPER_EMULATE_PUSH_DESCRIPTOR") && atoi(getenv("WRAPPER_EMULATE_PUSH_DESCRIPTOR")))
        ? "ON (forced)" :
     (pdev->vk.supported_extensions.KHR_push_descriptor && !pdev->base_supported_extensions.KHR_push_descriptor)
        ? "ON (base lacks it)" :
     pdev->base_supported_extensions.KHR_push_descriptor ? "off (native)" : "off");
   D("--- vkCreateDevice ---\n");
   D("  result: %d (%s)\n", result, result == VK_SUCCESS ? "VK_SUCCESS" : "FAILED");
   D("  requested device extensions (%u):\n", ci ? ci->enabledExtensionCount : 0);
   if (ci)
      for (uint32_t i = 0; i < ci->enabledExtensionCount; i++)
         D("    %s\n", ci->ppEnabledExtensionNames[i]);
   D("NOTE: seeing this block means a VkDevice was created. If a game fails\n");
   D("  and this block never appears, the D3D layer rejected caps BEFORE\n");
   D("  device creation -- the VKD3D_DEBUG/DXVK logs will show which.\n");
   D("full diagnosis also needs (app-side): VKD3D_DEBUG=warn DXVK_LOG_LEVEL=info WINEDEBUG=+vulkan\n");
   D("====================================================\n");

#undef D
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDevice(VkPhysicalDevice physicalDevice,
                     const VkDeviceCreateInfo* pCreateInfo,
                     const VkAllocationCallbacks* pAllocator,
                     VkDevice* pDevice)
{
   VK_FROM_HANDLE(wrapper_physical_device, physical_device, physicalDevice);
   wrapper_diag_autostart(physical_device);
   const char *wrapper_enable_extensions[VK_DEVICE_EXTENSION_COUNT];
   uint32_t wrapper_enable_extension_count = 0;
   VkDeviceCreateInfo wrapper_create_info = *pCreateInfo;
   struct vk_device_dispatch_table dispatch_table;
   struct wrapper_device *device;
   VkPhysicalDeviceFeatures2 *pdf2;
   VkPhysicalDeviceFeatures *pdf;
   VkResult result;
   static int wrapper_safe_create_device = -1;
   static int wrapper_device_fault = -1;
   VkPhysicalDeviceFaultFeaturesEXT fault_features_ext = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
   };
   bool used_fallback_create = false;
   VkDeviceQueueCreateInfo *query_reset_queues = NULL;
   float *query_reset_priorities = NULL;
   VkDeviceQueueCreateInfo *bcn_queues = NULL;
   float *bcn_priorities = NULL;

   device = vk_zalloc2(&physical_device->instance->vk.alloc, pAllocator,
                       sizeof(*device), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);

   list_inithead(&device->command_buffer_list);
   list_inithead(&device->device_memory_list);
   list_inithead(&device->image_list);
   list_inithead(&device->buffer_list);
   list_inithead(&device->fence_list);
   device->image_table = _mesa_hash_table_u64_create(NULL);
   device->image_view_table = _mesa_hash_table_u64_create(NULL);
   device->imageless_fb_table = _mesa_hash_table_u64_create(NULL);
   device->dynamic_pipeline_table = _mesa_hash_table_u64_create(NULL);
   device->stride_pipeline_table = _mesa_hash_table_u64_create(NULL);
   device->divisor_pipeline_table = _mesa_hash_table_u64_create(NULL);
   device->shader_code_table = _mesa_hash_table_u64_create(NULL);
   device->buffer_table = _mesa_hash_table_u64_create(NULL);
   device->device_only_memory = _mesa_hash_table_u64_create(NULL);
   device->fence_table = _mesa_hash_table_u64_create(NULL);
   
   simple_mtx_init(&device->resource_mutex, mtx_plain);
   simple_mtx_init(&device->bcn_gpu_mutex, mtx_plain);
   simple_mtx_init(&device->query_reset_mutex, mtx_plain);
   device->physical = physical_device;

   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wrapper_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wrapper_device_trampolines, false);

   result = vk_device_init(&device->vk, &physical_device->vk,
                           &dispatch_table, pCreateInfo, pAllocator);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to init Vulkan device, res %d", result);
      vk_free2(&physical_device->instance->vk.alloc, pAllocator,
               device);
      return vk_error(physical_device, result);
   }

   wrapper_filter_enabled_extensions(device,
      &wrapper_enable_extension_count, wrapper_enable_extensions);
   wrapper_append_required_extensions(device,
      &wrapper_enable_extension_count, wrapper_enable_extensions);

   /* VK_EXT_device_fault turns the generic VK_ERROR_DEVICE_LOST into an actual
    * GPU fault report (faulting address + vendor fault codes) that we dump in
    * QueueSubmit. It's universally available on Mali and cheap when no fault
    * occurs, so enable it by default whenever the base driver supports it;
    * WRAPPER_DEVICE_FAULT=0 opts out. */
   if (wrapper_device_fault == -1)
      wrapper_device_fault = getenv("WRAPPER_DEVICE_FAULT")
         ? atoi(getenv("WRAPPER_DEVICE_FAULT")) : 1;

   bool enable_device_fault = wrapper_device_fault &&
      physical_device->base_supported_extensions.EXT_device_fault;

   if (enable_device_fault)
      wrapper_enable_extensions[wrapper_enable_extension_count++] = "VK_EXT_device_fault";

   /* The BCn decode/encode shaders address their buffers by device address. */
   const bool base12 = physical_device->properties2.properties.apiVersion >= VK_API_VERSION_1_2;
   /* An application on a Vulkan 1.0/1.1 instance (DXVK 1.10) gets no core 1.2
    * entry points, so vkGetBufferDeviceAddress comes from the extension. */
   const bool app12 = physical_device->instance->vk.app_info.api_version >= VK_API_VERSION_1_2;
   const bool bcn_bda = physical_device->emulate_bcn > 1 &&
      physical_device->base_supported_features.bufferDeviceAddress &&
      ((base12 && app12) || physical_device->base_supported_extensions.KHR_buffer_device_address);
   if (bcn_bda && !(base12 && app12)) {
      bool listed = false;
      for (uint32_t i = 0; i < wrapper_enable_extension_count; i++)
         listed |= !strcmp(wrapper_enable_extensions[i], "VK_KHR_buffer_device_address");
      if (!listed)
         wrapper_enable_extensions[wrapper_enable_extension_count++] =
            "VK_KHR_buffer_device_address";
   }

   /* Policy bc6h=4x4: BC6H is stored as ASTC 4x4 HDR when the base driver has
    * textureCompressionASTC_HDR (core 1.3 or the EXT). */
   const bool base13 = physical_device->properties2.properties.apiVersion >= VK_API_VERSION_1_3;
   const bool bc6h_hdr = bcn_policy_bc6h_hdr() && physical_device->emulate_bcn > 1 &&
      physical_device->base_supported_features.textureCompressionASTC_HDR &&
      (base13 || physical_device->base_supported_extensions.EXT_texture_compression_astc_hdr);
   if (bcn_policy_bc6h_hdr() && !bc6h_hdr)
      WRAPPER_LOG(info, "BCn: no ASTC HDR on this device, BC6H is decoded");
   if (bc6h_hdr && !base13) {
      bool listed = false;
      for (uint32_t i = 0; i < wrapper_enable_extension_count; i++)
         listed |= !strcmp(wrapper_enable_extensions[i], "VK_EXT_texture_compression_astc_hdr");
      if (!listed)
         wrapper_enable_extensions[wrapper_enable_extension_count++] =
            "VK_EXT_texture_compression_astc_hdr";
   }

   wrapper_create_info.enabledExtensionCount = wrapper_enable_extension_count;
   wrapper_create_info.ppEnabledExtensionNames = wrapper_enable_extensions;
   
   pdf = (void *)pCreateInfo->pEnabledFeatures;
   pdf2 = __vk_find_struct((void *)pCreateInfo->pNext,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
            
#define DISABLE_FEATURE(f) \
if (pdf && pdf->f) { \
   pdf->f &= physical_device->base_supported_features.f; \
} \
\
if (pdf2 && pdf2->features.f) { \
   pdf2->features.f &= physical_device->base_supported_features.f; \
}

   DISABLE_FEATURE(textureCompressionBC);
   DISABLE_FEATURE(multiViewport);
   DISABLE_FEATURE(depthClamp);
   DISABLE_FEATURE(depthBiasClamp);
   DISABLE_FEATURE(fillModeNonSolid);
   DISABLE_FEATURE(shaderClipDistance);
   DISABLE_FEATURE(shaderCullDistance);
   DISABLE_FEATURE(dualSrcBlend);
   DISABLE_FEATURE(multiDrawIndirect);
   DISABLE_FEATURE(shaderInt64);

#undef DISABLE_FEATURE

   process_pnext_chain((VkBaseInStructure *)&wrapper_create_info, device->physical);
   wrapper_mask_feature_chain(physical_device,
                              (VkPhysicalDeviceFeatures *)wrapper_create_info.pEnabledFeatures,
                              (void *)wrapper_create_info.pNext);

   void *core13_storage = NULL;
   if (physical_device->emulate_vulkan13) {
      core13_storage = wrapper_core13_lower_device_create(device, &wrapper_create_info,
         wrapper_enable_extensions, &wrapper_enable_extension_count);
      wrapper_create_info.enabledExtensionCount = wrapper_enable_extension_count;
   }

   device->emulate_eds = physical_device->emulate_vulkan13 &&
      !physical_device->base_supported_extensions.EXT_extended_dynamic_state &&
      debug_get_bool_option("WRAPPER_EMULATE_EXTENDED_DYNAMIC_STATE", true);
   device->emulate_eds2 = physical_device->emulate_vulkan13 &&
      !physical_device->base_supported_extensions.EXT_extended_dynamic_state2 &&
      debug_get_bool_option("WRAPPER_EMULATE_EXTENDED_DYNAMIC_STATE", true);
   device->max_viewports =
      MAX2(physical_device->properties2.properties.limits.maxViewports, 1);
   if (device->emulate_eds || device->emulate_eds2)
      WRAPPER_LOG(info, "Emulating extended dynamic state%s with pipeline variants",
                  device->emulate_eds ? (device->emulate_eds2 ? " 1 and 2" : " 1") : " 2");

   device->fold_sets = physical_device->fold_sets;
   if (device->fold_sets)
      wrapper_fold_init(device);

   device->emulate_dynamic_stride = device->emulate_eds ||
      debug_get_bool_option("WRAPPER_EMULATE_DYNAMIC_STRIDE",
      physical_device->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY &&
      VK_API_VERSION_MAJOR(physical_device->properties2.properties.driverVersion) < 44);
   if (device->emulate_dynamic_stride)
      WRAPPER_LOG(info, "Emulating dynamic vertex input binding strides");

   device->emulate_vertex_divisor = debug_get_bool_option("WRAPPER_EMULATE_VERTEX_DIVISOR",
      physical_device->emulate_vertex_divisor);
   device->emulate_zero_divisor = !device->emulate_vertex_divisor &&
      physical_device->emulate_zero_divisor;
   if (device->emulate_vertex_divisor)
      WRAPPER_LOG(info, "Emulating vertex attribute divisors");
   else if (device->emulate_zero_divisor)
      WRAPPER_LOG(info, "Emulating the zero vertex attribute divisor");

   device->cmdbuf_entry_barrier = debug_get_bool_option("WRAPPER_CMDBUF_ENTRY_BARRIER",
      physical_device->driver_properties.driverID == VK_DRIVER_ID_IMAGINATION_PROPRIETARY);

   VkPhysicalDeviceTextureCompressionASTCHDRFeatures astc_hdr_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES,
      .textureCompressionASTC_HDR = VK_TRUE,
   };
   if (bc6h_hdr) {
      VkPhysicalDeviceVulkan13Features *v13 = (void *)vk_find_struct_const(
         wrapper_create_info.pNext, PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
      VkPhysicalDeviceTextureCompressionASTCHDRFeatures *hf = (void *)vk_find_struct_const(
         wrapper_create_info.pNext, PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES);
      if (v13)
         v13->textureCompressionASTC_HDR = VK_TRUE;
      else if (hf)
         hf->textureCompressionASTC_HDR = VK_TRUE;
      else {
         astc_hdr_features.pNext = (void *)wrapper_create_info.pNext;
         wrapper_create_info.pNext = &astc_hdr_features;
      }
   }

   VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .bufferDeviceAddress = VK_TRUE,
   };
   if (bcn_bda) {
      VkPhysicalDeviceVulkan12Features *v12 = (void *)vk_find_struct_const(
         wrapper_create_info.pNext, PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
      VkPhysicalDeviceBufferDeviceAddressFeatures *bf = (void *)vk_find_struct_const(
         wrapper_create_info.pNext, PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES);
      if (v12)
         v12->bufferDeviceAddress = VK_TRUE;
      else if (bf)
         bf->bufferDeviceAddress = VK_TRUE;
      else {
         bda_features.pNext = (void *)wrapper_create_info.pNext;
         wrapper_create_info.pNext = &bda_features;
      }
   }

   /* Request the deviceFault feature. Only inject our struct if the client
    * didn't already provide one (it manages its own if so). */
   if (enable_device_fault &&
       !vk_find_struct_const(wrapper_create_info.pNext, PHYSICAL_DEVICE_FAULT_FEATURES_EXT)) {
      WRAPPER_LOG(info, "Enabling VK_EXT_device_fault for GPU fault reporting");
      fault_features_ext.deviceFault =
         physical_device->base_supported_features.deviceFault;
      fault_features_ext.deviceFaultVendorBinary =
         physical_device->base_supported_features.deviceFaultVendorBinary;
      fault_features_ext.pNext = (void *)wrapper_create_info.pNext;
      wrapper_create_info.pNext = &fault_features_ext;
   }

   if (WRAPPER_LOG_LEVEL(info)) {
      for (int i = 0; i < wrapper_enable_extension_count; i++) {
         WRAPPER_LOG(info, "Enabling device extension %s", wrapper_enable_extensions[i]);
      }
   }

   if (wrapper_safe_create_device == -1) {
      wrapper_safe_create_device = getenv("WRAPPER_SAFE_CREATE_DEVICE") ? atoi(getenv("WRAPPER_SAFE_CREATE_DEVICE")) : 0;
   }
   
   const bool emulate_query_reset = device->vk.enabled_features.hostQueryReset &&
      !physical_device->base_supported_features.hostQueryReset;
   if (emulate_query_reset) {
      result = wrapper_query_reset_reserve_queue(device, &wrapper_create_info,
                                                 &query_reset_queues,
                                                 &query_reset_priorities);
      if (result != VK_SUCCESS) {
         wrapper_DestroyDevice(wrapper_device_to_handle(device), &device->vk.alloc);
         return vk_error(physical_device, result);
      }
   }

   if (physical_device->emulate_bcn &&
       !(emulate_query_reset && !device->query_reset_shared_queue))
      wrapper_bcn_reserve_queue(device, &wrapper_create_info, &bcn_queues, &bcn_priorities);

   result = physical_device->dispatch_table.CreateDevice(
      physical_device->dispatch_handle, &wrapper_create_info,
         pAllocator, &device->dispatch_handle);
   if (wrapper_diag_capture)
      wrapper_diag_append("[WRAPPER_DIAG] primary vkCreateDevice result=%d "
                          "featureChain=%d safeCreate=%d\n", result,
                          wrapper_create_info.pNext != NULL, wrapper_safe_create_device);
   wrapper_core13_free_device_create(core13_storage);

   if (result != VK_SUCCESS) {
      if (wrapper_safe_create_device) {
         WRAPPER_LOG(error, "Driver vkCreateDevice failed (%d) with the feature chain, "
                            "retrying with a NULL pNext chain: every requested "
                            "extension feature is dropped", result);
         WRAPPER_LOG(info, "Forcing device creation with a NULL pNext chain");
         wrapper_diag_append("[BCN] vkCreateDevice returned %d, retrying without the "
                             "feature chain\n", result);
         wrapper_create_info.pNext = NULL;
         used_fallback_create = true;
         result = physical_device->dispatch_table.CreateDevice(
            physical_device->dispatch_handle, &wrapper_create_info,
               pAllocator, &device->dispatch_handle);
         if (wrapper_diag_capture)
            wrapper_diag_append("[WRAPPER_DIAG] NULL-chain retry result=%d\n", result);
      }
      
      if (result != VK_SUCCESS) {
         free(query_reset_queues);
         free(query_reset_priorities);
         free(bcn_queues);
         free(bcn_priorities);
         WRAPPER_LOG(error, "Failed driver createDevice, res %d", result);
         wrapper_emit_diag(physical_device, pCreateInfo, result);
         wrapper_DestroyDevice(wrapper_device_to_handle(device),
                               &device->vk.alloc);
         return vk_error(physical_device, result);
      }
   }

   free(query_reset_queues);
   free(query_reset_priorities);
   free(bcn_queues);
   free(bcn_priorities);

   void *gdpa = physical_device->instance->dispatch_table.GetInstanceProcAddr(
      physical_device->instance->dispatch_handle, "vkGetDeviceProcAddr");
   vk_device_dispatch_table_load(&device->dispatch_table, gdpa,
                                 device->dispatch_handle);
   if (physical_device->emulate_vulkan13)
      wrapper_lower_missing_commands(device);

   if (emulate_query_reset) {
      result = wrapper_query_reset_init(device);
      if (result != VK_SUCCESS) {
         wrapper_DestroyDevice(wrapper_device_to_handle(device), &device->vk.alloc);
         return vk_error(physical_device, result);
      }
   }

   /* The fallback create with a NULL pNext drops the deviceFault feature, so
    * only treat fault reporting as usable when the primary create succeeded. */
   device->device_fault_enabled = enable_device_fault && !used_fallback_create;

   device->emulate_null_descriptor =
      physical_device->vk.supported_extensions.EXT_robustness2 &&
      !physical_device->base_supported_features.nullDescriptor;
   if (device->emulate_null_descriptor) {
      WRAPPER_LOG(info, "Emulating nullDescriptor with canonical zero resources");
      wrapper_create_null_resources(device);
   }

   device->emulate_imageless_framebuffer =
      physical_device->emulate_imageless_framebuffer &&
      device->vk.enabled_extensions.KHR_imageless_framebuffer;
   if (device->emulate_imageless_framebuffer)
      WRAPPER_LOG(info, "Emulating imageless framebuffers for this device");

   /* Push-descriptor emulation: on when the app enabled VK_KHR_push_descriptor
    * and either the base driver lacks it or WRAPPER_EMULATE_PUSH_DESCRIPTOR
    * forces it (so it can be validated on a device that has it natively). */
   {
      static int force = -1;
      if (force == -1)
         force = getenv("WRAPPER_EMULATE_PUSH_DESCRIPTOR")
            ? atoi(getenv("WRAPPER_EMULATE_PUSH_DESCRIPTOR")) : 0;
      bool app_wants = device->vk.enabled_extensions.KHR_push_descriptor;
      bool base_has = physical_device->base_supported_extensions.KHR_push_descriptor;
      device->emulate_push_descriptor = app_wants && (force || !base_has);
      if (device->emulate_push_descriptor) {
         WRAPPER_LOG(info, "Emulating VK_KHR_push_descriptor%s",
                     (base_has && force) ? " (forced; base has it natively)" : "");
         device->max_push_descriptors = WRAPPER_MAX_PUSH_DESCRIPTORS;
         simple_mtx_init(&device->push_mutex, mtx_plain);
         device->push_dsl_table = _mesa_hash_table_u64_create(NULL);
         device->push_pl_table = _mesa_hash_table_u64_create(NULL);
         device->push_template_table = _mesa_hash_table_u64_create(NULL);
      }
   }

   result = wrapper_create_device_queue(device, pCreateInfo);
   if (result != VK_SUCCESS) {
      wrapper_DestroyDevice(wrapper_device_to_handle(device),
                            &device->vk.alloc);
      return vk_error(physical_device, result);
   }

   if (!physical_device->vk.supported_features.memoryMapPlaced) {
      device->vk.dispatch_table.AllocateMemory =
         wrapper_device_trampolines.AllocateMemory;
      device->vk.dispatch_table.MapMemory2 =
         wrapper_device_trampolines.MapMemory2;
      device->vk.dispatch_table.UnmapMemory =
         wrapper_device_trampolines.UnmapMemory;
      device->vk.dispatch_table.UnmapMemory2 =
         wrapper_device_trampolines.UnmapMemory2;
      device->vk.dispatch_table.FreeMemory =
         wrapper_device_trampolines.FreeMemory;
   }

   device->bcn_bda = bcn_bda && !used_fallback_create &&
                     device->dispatch_table.GetBufferDeviceAddress;
   bcn_set_astc_hdr(bc6h_hdr && !used_fallback_create);
   if (physical_device->emulate_bcn > 1) {
      char storage[512];
      bcn_describe_storage(storage, sizeof(storage));
      wrapper_diag_append("[BCN] storage %s\n", storage);
      wrapper_diag_append("[BCN] bufferDeviceAddress: base=%d api=%u.%u app=%u.%u requested=%d "
                          "fallback=%d\n",
                          physical_device->base_supported_features.bufferDeviceAddress,
                          VK_API_VERSION_MAJOR(physical_device->properties2.properties.apiVersion),
                          VK_API_VERSION_MINOR(physical_device->properties2.properties.apiVersion),
                          VK_API_VERSION_MAJOR(physical_device->instance->vk.app_info.api_version),
                          VK_API_VERSION_MINOR(physical_device->instance->vk.app_info.api_version),
                          bcn_bda, used_fallback_create);
      wrapper_bcn_gpu_init(device, pCreateInfo->queueCreateInfoCount ?
         pCreateInfo->pQueueCreateInfos[0].queueFamilyIndex : UINT32_MAX);
      wrapper_bcn_gpu_prebuild(device);
   }
   if (bc6h_hdr && !used_fallback_create)
      WRAPPER_LOG(info, "BCn policy: BC6H stored as ASTC 4x4 HDR");
   wrapper_emit_diag(physical_device, pCreateInfo, VK_SUCCESS);

   *pDevice = wrapper_device_to_handle(device);

   return VK_SUCCESS;
}

static void 
wrapper_buffer_destroy(struct wrapper_device *device,
					   struct wrapper_buffer *wb,
					   const VkAllocationCallbacks *pAllocator)
{
   if (wb == NULL)
      return;

   simple_mtx_lock(&device->resource_mutex);
      
   device->dispatch_table.DestroyBuffer(device->dispatch_handle,
      wb->dispatch_handle, pAllocator);

   _mesa_hash_table_u64_remove(device->buffer_table, (uint64_t)wb->dispatch_handle);
   list_del(&wb->link);

   simple_mtx_unlock(&device->resource_mutex);
   
   vk_object_free(&device->vk, &device->vk.alloc, wb);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateBuffer(VkDevice _device,
					 const VkBufferCreateInfo *pCreateInfo,
					 const VkAllocationCallbacks *pAllocator,
					 VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;
   VkExternalMemoryHandleTypeFlags handle_types = 0;

   /* When we fake VK_KHR_maintenance5 (base driver lacks it, e.g. Xclipse),
    * DXVK specifies buffer usage through VkBufferUsageFlags2CreateInfo and may
    * leave the 32-bit VkBufferCreateInfo::usage as 0. The base driver ignores
    * the unknown pNext struct, so fold the flags2 usage back into the 32-bit
    * field before forwarding. */
   VkBufferCreateInfo local_create_info;
   if (!device->physical->base_supported_extensions.KHR_maintenance5) {
      const VkBufferUsageFlags2CreateInfo *uf2 = __vk_find_struct(
         (void *)pCreateInfo->pNext, VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO);
      if (uf2) {
         local_create_info = *pCreateInfo;
         local_create_info.usage |= (VkBufferUsageFlags)uf2->usage;
         pCreateInfo = &local_create_info;
      }
   }

   const VkExternalMemoryBufferCreateInfo *ext_info =
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   if (ext_info) {
      handle_types = ext_info->handleTypes;
   }

   res = device->dispatch_table.CreateBuffer(device->dispatch_handle,
      pCreateInfo, pAllocator, pBuffer);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create buffer, res %d", res);
      return res;
   }

   simple_mtx_lock(&device->resource_mutex);

   struct wrapper_buffer *wb = vk_object_zalloc(&device->vk, 
      &device->vk.alloc, sizeof(struct wrapper_buffer), VK_OBJECT_TYPE_BUFFER);

   if (!wb) {
      WRAPPER_LOG(error, "Failed to allocate wrapper_buffer");
      simple_mtx_unlock(&device->resource_mutex);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
      
   wb->device = device;
   wb->size = pCreateInfo->size;
   wb->dispatch_handle = *pBuffer;
   wb->handle_types = handle_types;

   list_add(&wb->link, &device->buffer_list);
   _mesa_hash_table_u64_insert(device->buffer_table, (uint64_t)wb->dispatch_handle, wb);

   simple_mtx_unlock(&device->resource_mutex);
   
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_BindBufferMemory(VkDevice _device,
						 VkBuffer buffer,
						 VkDeviceMemory memory,
						 VkDeviceSize memoryOffset)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;

   res = device->dispatch_table.BindBufferMemory(device->dispatch_handle,
      buffer, memory, memoryOffset);
   
   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to bind buffer memory, res %d", res);
      return res;
   }

   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle(device, buffer);
   if (wb == NULL) {
      WRAPPER_LOG(error, "Failed to query wrapper_buffer");
      simple_mtx_unlock(&device->resource_mutex);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   wb->memory = memory;
   wb->offset = memoryOffset;
   wb->device_only = wrapper_memory_device_only(device, memory);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_BindBufferMemory2(VkDevice _device,
                          uint32_t bindInfoCount,
                          const VkBindBufferMemoryInfo *pBindInfos)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;

   res = device->dispatch_table.BindBufferMemory2(device->dispatch_handle,
      bindInfoCount, pBindInfos);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to bind buffer memory2, res %d", res);
      return res;
   }

   for (uint32_t i = 0; i < bindInfoCount; i++) {
      struct wrapper_buffer *wb =
         get_wrapper_buffer_from_handle(device, pBindInfos[i].buffer);
      if (wb) {
         wb->memory = pBindInfos[i].memory;
         wb->offset = pBindInfos[i].memoryOffset;
         wb->device_only = wrapper_memory_device_only(device, pBindInfos[i].memory);
      }
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyBuffer(VkDevice _device,
					  VkBuffer buffer,
					  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle(device, buffer);
   wrapper_buffer_destroy(device, wb, pAllocator);
}

static void 
wrapper_image_destroy(struct wrapper_device *device,
					  struct wrapper_image *wi,
					  const VkAllocationCallbacks *pAllocator)
{
   if (wi == NULL)
      return;

   simple_mtx_lock(&device->resource_mutex);
      
   device->dispatch_table.DestroyImage(device->dispatch_handle,
      wi->dispatch_handle, pAllocator);

   _mesa_hash_table_u64_remove(device->image_table, (uint64_t)wi->dispatch_handle);
   list_del(&wi->link);
   if (wi->mip_drop)
      __atomic_sub_fetch(&wrapper_capped_images, 1, __ATOMIC_RELAXED);

   simple_mtx_unlock(&device->resource_mutex);
   
   vk_object_free(&device->vk, &device->vk.alloc, wi);
}

struct bcn_view_format_list {
   VkImageFormatListCreateInfo list;
   VkFormat inline_formats[8];
   VkFormat *formats;
   union {
      VkBaseInStructure base;
      VkExternalMemoryImageCreateInfo external;
      VkImageStencilUsageCreateInfo stencil;
      VkImageSwapchainCreateInfoKHR swapchain;
      VkImageDrmFormatModifierListCreateInfoEXT modifier_list;
      VkImageDrmFormatModifierExplicitCreateInfoEXT modifier_explicit;
      struct wsi_image_create_info wsi;
   } nodes[4];
};

static size_t
bcn_chain_node_size(VkStructureType type)
{
   switch ((int)type) {
   case VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO:
      return sizeof(VkExternalMemoryImageCreateInfo);
   case VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO:
      return sizeof(VkImageStencilUsageCreateInfo);
   case VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR:
      return sizeof(VkImageSwapchainCreateInfoKHR);
   case VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT:
      return sizeof(VkImageDrmFormatModifierListCreateInfoEXT);
   case VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT:
      return sizeof(VkImageDrmFormatModifierExplicitCreateInfoEXT);
   case VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA:
      return sizeof(struct wsi_image_create_info);
   default:
      return 0;
   }
}

static const void *
bcn_substitute_view_formats(struct wrapper_physical_device *pdev,
                            const void *chain, struct bcn_view_format_list *out)
{
   const VkBaseInStructure *node = NULL;
   unsigned depth = 0;

   out->formats = NULL;
   for (const VkBaseInStructure *s = chain; s; s = s->pNext) {
      if (s->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
         node = s;
         break;
      }
      depth++;
   }
   if (!node)
      return chain;

   const VkImageFormatListCreateInfo *fl = (const VkImageFormatListCreateInfo *)node;
   out->formats = fl->viewFormatCount <= ARRAY_SIZE(out->inline_formats) ?
      out->inline_formats : malloc(fl->viewFormatCount * sizeof(VkFormat));
   if (!out->formats)
      return chain;

   for (uint32_t i = 0; i < fl->viewFormatCount; i++) {
      out->formats[i] = is_emulated_bcn(pdev, fl->pViewFormats[i]) ?
         get_format_for_bcn(fl->pViewFormats[i]) : fl->pViewFormats[i];
      if (wrapper_diag_on() && out->formats[i] != fl->pViewFormats[i])
         wrapper_diag_append("[FMT] CreateImage view format[%u] %d -> %d (copy)\n",
            i, fl->pViewFormats[i], out->formats[i]);
   }
   out->list = *fl;
   out->list.pViewFormats = out->formats;

   const VkBaseInStructure *s = chain;
   bool copied = depth <= ARRAY_SIZE(out->nodes);
   for (unsigned i = 0; copied && i < depth; i++, s = s->pNext) {
      size_t size = bcn_chain_node_size(s->sType);
      if (!size)
         copied = false;
      else
         memcpy(&out->nodes[i], s, size);
   }

   if (!copied) {
      if (wrapper_diag_on())
         wrapper_diag_append("[FMT] CreateImage view format list behind an unknown struct, prepending copy\n");
      out->list.pNext = chain;
      return &out->list;
   }

   for (unsigned i = 0; i < depth; i++)
      out->nodes[i].base.pNext = i + 1 < depth ?
         &out->nodes[i + 1].base : (const VkBaseInStructure *)&out->list;
   return depth ? (const void *)&out->nodes[0] : (const void *)&out->list;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateImage(VkDevice _device,
					const VkImageCreateInfo *pCreateInfo,
					const VkAllocationCallbacks *pAllocator,
					VkImage *pImage)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;
   VkImageCreateInfo create_info;
   bool is_emulated_bgra8 = false;
   bool is_wsi_image = false;
   VkExternalMemoryHandleTypeFlags handle_types = 0;
   uint32_t mip_drop = 0;
   struct bcn_view_format_list view_formats = { .formats = NULL };

   // Wrapper specific extension for B8G8R8A8 AHB img emulation for the swapchain
   VkBaseInStructure *prev = (VkBaseInStructure *) pCreateInfo;
   for (const VkBaseInStructure *s = pCreateInfo->pNext; s; s = s->pNext) {
      if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EMULATED_B8G8R8A8_CREATE_INFO_EXT) {
         is_emulated_bgra8 = true;
         prev->pNext = s->pNext; // unlink
         break;
      }
      prev = (VkBaseInStructure *) s;
   }

   // Tag swapchain images using VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA
   const struct wsi_image_create_info *wsi_info =
      vk_find_struct_const(pCreateInfo->pNext, WSI_IMAGE_CREATE_INFO_MESA);
   if (wsi_info) {
      is_wsi_image = true;
   }

   const VkExternalMemoryImageCreateInfo *ext_info =
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   if (ext_info) {
      handle_types = ext_info->handleTypes;
   }

   /* Copy after the unlink above: when the emulated-bgra8 struct is the
    * head of the pNext chain, the unlink only updates pCreateInfo->pNext,
    * so a copy taken earlier would still pass the unknown struct to the
    * driver. */
   create_info = *pCreateInfo;

   if (is_emulated_bcn(device->physical, pCreateInfo->format)) {
      wrapper_bcn_gpu_storage_check(device, pCreateInfo->format);
      create_info.format = get_format_for_bcn(pCreateInfo->format);
      if (create_info.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)
         create_info.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

      if (wrapper_diag_on())
         wrapper_diag_append(
            "[FMT] CreateImage fmt=%d -> %d flags=0x%x usage=0x%x %ux%u mips=%u layers=%u\n",
            pCreateInfo->format, create_info.format, pCreateInfo->flags,
            pCreateInfo->usage, pCreateInfo->extent.width,
            pCreateInfo->extent.height, pCreateInfo->mipLevels,
            pCreateInfo->arrayLayers);

      create_info.pNext = bcn_substitute_view_formats(device->physical,
         create_info.pNext, &view_formats);

      if (!handle_types && !is_wsi_image)
         mip_drop = bcn_cap_mip_drop(pCreateInfo);
      if (mip_drop) {
         create_info.extent.width = MAX2(1, pCreateInfo->extent.width >> mip_drop);
         create_info.extent.height = MAX2(1, pCreateInfo->extent.height >> mip_drop);
         create_info.mipLevels = pCreateInfo->mipLevels - mip_drop;
      }
   }

   res = device->dispatch_table.CreateImage(device->dispatch_handle,
      &create_info, pAllocator, pImage);

   if (view_formats.formats != view_formats.inline_formats)
      free(view_formats.formats);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create image, res %d", res);
      return res;
   }

   simple_mtx_lock(&device->resource_mutex);

   struct wrapper_image *wi = vk_object_zalloc(&device->vk,
      &device->vk.alloc, sizeof(struct wrapper_image), VK_OBJECT_TYPE_IMAGE);

   if (!wi) {
      WRAPPER_LOG(error, "Failed to allocate wrapper_image");
      simple_mtx_unlock(&device->resource_mutex);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   wi->device = device;
   wi->info = *pCreateInfo;
   wi->dispatch_handle = *pImage;
   wi->is_emulated_bgra8 = is_emulated_bgra8;
   wi->is_wsi_image = is_wsi_image;
   wi->handle_types = handle_types;
   wi->mip_drop = mip_drop;

   list_add(&wi->link, &device->image_list);
   _mesa_hash_table_u64_insert(device->image_table, (uint64_t)wi->dispatch_handle, wi);
   if (mip_drop)
      __atomic_add_fetch(&wrapper_capped_images, 1, __ATOMIC_RELAXED);

   simple_mtx_unlock(&device->resource_mutex);

   if (mip_drop)
      wrapper_diag_append(
         "[CAP] img=%04x fmt=%d %ux%u mips=%u -> %ux%u mips=%u (dropped %u)\n",
         (unsigned)((uintptr_t)*pImage & 0xffff), pCreateInfo->format,
         pCreateInfo->extent.width, pCreateInfo->extent.height,
         pCreateInfo->mipLevels, create_info.extent.width,
         create_info.extent.height, create_info.mipLevels, mip_drop);

   return VK_SUCCESS;
}

static void
wrapper_device_image_memory_requirements(struct wrapper_device *device,
      const VkDeviceImageMemoryRequirements *pInfo,
      VkMemoryRequirements2 *pMemoryRequirements)
{
   VkDeviceImageMemoryRequirements info = *pInfo;
   VkImageCreateInfo ci;

   if (pInfo->pCreateInfo &&
       is_emulated_bcn(device->physical, pInfo->pCreateInfo->format)) {
      ci = *pInfo->pCreateInfo;
      wrapper_bcn_gpu_storage_check(device, pInfo->pCreateInfo->format);
      ci.format = get_format_for_bcn(pInfo->pCreateInfo->format);
      ci.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      ci.pNext = NULL;
      uint32_t mip_drop = vk_find_struct_const(pInfo->pCreateInfo->pNext,
         EXTERNAL_MEMORY_IMAGE_CREATE_INFO) ? 0 : bcn_cap_mip_drop(pInfo->pCreateInfo);
      if (mip_drop) {
         ci.extent.width = MAX2(1, ci.extent.width >> mip_drop);
         ci.extent.height = MAX2(1, ci.extent.height >> mip_drop);
         ci.mipLevels -= mip_drop;
      }
      info.pCreateInfo = &ci;
   }

   if (device->dispatch_table.GetDeviceImageMemoryRequirements) {
      device->dispatch_table.GetDeviceImageMemoryRequirements(device->dispatch_handle,
         &info, pMemoryRequirements);
      return;
   }

   VkImage image;
   if (device->dispatch_table.CreateImage(device->dispatch_handle, info.pCreateInfo,
                                          NULL, &image) != VK_SUCCESS) {
      pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){ 0 };
      return;
   }
   VkImageMemoryRequirementsInfo2 image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = image,
   };
   device->dispatch_table.GetImageMemoryRequirements2(device->dispatch_handle,
      &image_info, pMemoryRequirements);
   device->dispatch_table.DestroyImage(device->dispatch_handle, image, NULL);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceBufferMemoryRequirements(VkDevice _device,
      const VkDeviceBufferMemoryRequirements *pInfo,
      VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   if (device->dispatch_table.GetDeviceBufferMemoryRequirements) {
      device->dispatch_table.GetDeviceBufferMemoryRequirements(device->dispatch_handle,
         pInfo, pMemoryRequirements);
      return;
   }

   VkBufferCreateInfo create_info = *pInfo->pCreateInfo;
   const VkBufferUsageFlags2CreateInfo *usage2 = __vk_find_struct(
      (void *)create_info.pNext, VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO);
   if (usage2 && !device->physical->base_supported_extensions.KHR_maintenance5)
      create_info.usage |= (VkBufferUsageFlags)usage2->usage;

   VkBuffer buffer;
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &create_info,
                                           NULL, &buffer) != VK_SUCCESS) {
      pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){ 0 };
      return;
   }
   VkBufferMemoryRequirementsInfo2 buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
      .buffer = buffer,
   };
   device->dispatch_table.GetBufferMemoryRequirements2(device->dispatch_handle,
      &buffer_info, pMemoryRequirements);
   device->dispatch_table.DestroyBuffer(device->dispatch_handle, buffer, NULL);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceImageSparseMemoryRequirements(VkDevice _device,
      const VkDeviceImageMemoryRequirements *pInfo,
      uint32_t *pSparseMemoryRequirementCount,
      VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   if (device->dispatch_table.GetDeviceImageSparseMemoryRequirements) {
      device->dispatch_table.GetDeviceImageSparseMemoryRequirements(device->dispatch_handle,
         pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
      return;
   }

   *pSparseMemoryRequirementCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceImageMemoryRequirements(VkDevice _device,
      const VkDeviceImageMemoryRequirements *pInfo,
      VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   wrapper_device_image_memory_requirements(device, pInfo, pMemoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateImageView(VkDevice _device,
						const VkImageViewCreateInfo *pCreateInfo,
						const VkAllocationCallbacks *pAllocator,
						VkImageView *pView)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkImageViewCreateInfo create_info = *pCreateInfo;
   VkResult result;

   if (is_emulated_bcn(device->physical, pCreateInfo->format)) {
      create_info.format = get_format_for_bcn(pCreateInfo->format);
      bcn_view_swizzle(pCreateInfo->format, &create_info.components);
      if (wrapper_diag_on())
         wrapper_diag_append("[FMT] CreateImageView fmt=%d -> %d type=%d\n",
            pCreateInfo->format, create_info.format, pCreateInfo->viewType);
   }

   uint32_t mip_drop = wrapper_image_mip_drop(device, pCreateInfo->image);
   if (mip_drop && !bcn_cap_range(mip_drop, &create_info.subresourceRange)) {
      create_info.subresourceRange.baseMipLevel = 0;
      create_info.subresourceRange.levelCount = 1;
   }

   result = device->dispatch_table.CreateImageView(device->dispatch_handle,
     &create_info, pAllocator, pView);

   if (result != VK_SUCCESS)
      WRAPPER_LOG(error, "Failed to create image view, res %d", result);
   /* Only the lowering paths read this table, and populating it costs an
    * allocation and a locked hash insert per view, so do not pay it on a
    * driver that needs neither. */
   else if (device->physical->emulate_vulkan13 ||
            device->physical->emulate_imageless_framebuffer) {
      struct wrapper_image_view *view = calloc(1, sizeof(*view));
      if (view) {
         struct wrapper_image *image =
            get_wrapper_image_from_handle(device, pCreateInfo->image);
         view->handle = *pView;
         view->image = pCreateInfo->image;
         view->format = create_info.format;
         view->samples = image ? image->info.samples : VK_SAMPLE_COUNT_1_BIT;
         simple_mtx_lock(&device->resource_mutex);
         _mesa_hash_table_u64_insert(device->image_view_table,
                                     (uint64_t)*pView, view);
         simple_mtx_unlock(&device->resource_mutex);
      }
   }

   return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyImageView(VkDevice _device, VkImageView imageView,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_image_view *view = NULL;

   simple_mtx_lock(&device->resource_mutex);
   view = _mesa_hash_table_u64_search(device->image_view_table,
                                      (uint64_t)imageView);
   if (view)
      _mesa_hash_table_u64_remove(device->image_view_table,
                                  (uint64_t)imageView);
   simple_mtx_unlock(&device->resource_mutex);

   free(view);
   device->dispatch_table.DestroyImageView(device->dispatch_handle, imageView,
                                           pAllocator);
}

/* ---- VK_KHR_imageless_framebuffer emulation ---------------------------- *
 *
 * An imageless framebuffer names no image views at creation; they arrive at
 * vkCmdBeginRenderPass in VkRenderPassAttachmentBeginInfo.  A driver without
 * the extension can still be given a plain framebuffer at that moment, so the
 * emulation is deferral and nothing else: hold the dimensions, build the real
 * object when the views turn up, and drop it with the rest of the command
 * buffer's transient render objects.
 *
 * The handle handed to the application is the address of the record.  Every
 * lookup goes through imageless_fb_table, so a driver framebuffer is never
 * mistaken for one of ours.
 */
static uint64_t
wrapper_framebuffer_key(VkFramebuffer framebuffer)
{
   return (uint64_t)(uintptr_t)framebuffer;
}

static struct wrapper_imageless_framebuffer *
wrapper_lookup_imageless_framebuffer(struct wrapper_device *device,
                                     VkFramebuffer framebuffer)
{
   struct wrapper_imageless_framebuffer *fb;

   if (!device->emulate_imageless_framebuffer ||
       framebuffer == VK_NULL_HANDLE)
      return NULL;

   simple_mtx_lock(&device->resource_mutex);
   fb = _mesa_hash_table_u64_search(device->imageless_fb_table,
                                    wrapper_framebuffer_key(framebuffer));
   simple_mtx_unlock(&device->resource_mutex);

   return fb;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateFramebuffer(VkDevice _device,
                          const VkFramebufferCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkFramebuffer *pFramebuffer)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   if (!device->emulate_imageless_framebuffer ||
       !(pCreateInfo->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT))
      return device->dispatch_table.CreateFramebuffer(device->dispatch_handle,
                                                      pCreateInfo, pAllocator,
                                                      pFramebuffer);

   struct wrapper_imageless_framebuffer *fb = calloc(1, sizeof(*fb));
   if (!fb)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   fb->attachment_count = pCreateInfo->attachmentCount;
   fb->width = pCreateInfo->width;
   fb->height = pCreateInfo->height;
   fb->layers = pCreateInfo->layers;

   VkFramebuffer handle = (VkFramebuffer)(uintptr_t)fb;

   simple_mtx_lock(&device->resource_mutex);
   _mesa_hash_table_u64_insert(device->imageless_fb_table,
                               wrapper_framebuffer_key(handle), fb);
   simple_mtx_unlock(&device->resource_mutex);

   WRAPPER_LOG(info,
      "Imageless framebuffer %ux%u layers=%u attachments=%u",
      fb->width, fb->height, fb->layers, fb->attachment_count);

   *pFramebuffer = handle;
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyFramebuffer(VkDevice _device, VkFramebuffer framebuffer,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_imageless_framebuffer *fb = NULL;

   if (framebuffer == VK_NULL_HANDLE)
      return;

   if (device->emulate_imageless_framebuffer) {
      simple_mtx_lock(&device->resource_mutex);
      fb = _mesa_hash_table_u64_search(device->imageless_fb_table,
                                       wrapper_framebuffer_key(framebuffer));
      if (fb)
         _mesa_hash_table_u64_remove(device->imageless_fb_table,
                                     wrapper_framebuffer_key(framebuffer));
      simple_mtx_unlock(&device->resource_mutex);
   }

   if (fb) {
      free(fb);
      return;
   }

   device->dispatch_table.DestroyFramebuffer(device->dispatch_handle,
                                             framebuffer, pAllocator);
}

/* Returns true when the begin info names a wrapper-owned imageless
 * framebuffer, in which case *out holds the driver framebuffer to use --
 * VK_NULL_HANDLE if one could not be built, and then the render pass is
 * dropped rather than begun against a handle the driver never issued. */
static bool
wrapper_lower_imageless_framebuffer(struct wrapper_command_buffer *wcb,
                                    const VkRenderPassBeginInfo *pBegin,
                                    VkFramebuffer *out)
{
   struct wrapper_device *device = wcb->device;
   const VkRenderPassAttachmentBeginInfo *attachments = NULL;
   struct wrapper_imageless_framebuffer *fb;

   fb = wrapper_lookup_imageless_framebuffer(device, pBegin->framebuffer);
   if (!fb)
      return false;

   *out = VK_NULL_HANDLE;

   vk_foreach_struct_const(s, pBegin->pNext) {
      if (s->sType == VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO) {
         attachments = (const VkRenderPassAttachmentBeginInfo *)s;
         break;
      }
   }

   if (!attachments) {
      WRAPPER_LOG(error,
         "Imageless framebuffer begun without VkRenderPassAttachmentBeginInfo");
      return true;
   }
   if (attachments->attachmentCount != fb->attachment_count) {
      WRAPPER_LOG(error,
         "Imageless framebuffer attachment count %u, begin supplies %u",
         fb->attachment_count, attachments->attachmentCount);
      return true;
   }

   /* Built against the render pass of this begin, which the spec already
    * requires to be compatible with the one the framebuffer was created
    * with -- so the stricter of the two is the one actually in use. */
   VkFramebufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = pBegin->renderPass,
      .attachmentCount = attachments->attachmentCount,
      .pAttachments = attachments->pAttachments,
      .width = fb->width,
      .height = fb->height,
      .layers = fb->layers,
   };
   VkFramebuffer real = VK_NULL_HANDLE;
   VkResult result = device->dispatch_table.CreateFramebuffer(
      device->dispatch_handle, &info, NULL, &real);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to realise imageless framebuffer: %d", result);
      return true;
   }

   struct wrapper_dynamic_render_object *object = calloc(1, sizeof(*object));
   if (!object) {
      device->dispatch_table.DestroyFramebuffer(device->dispatch_handle, real,
                                                NULL);
      return true;
   }
   object->framebuffer = real;
   list_addtail(&object->link, &wcb->dynamic_render_objects);

   *out = real;
   return true;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBeginRenderPass(VkCommandBuffer commandBuffer,
                           const VkRenderPassBeginInfo *pRenderPassBegin,
                           VkSubpassContents contents)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   VkRenderPassBeginInfo begin = *pRenderPassBegin;
   VkFramebuffer real;

   if (wrapper_lower_imageless_framebuffer(wcb, pRenderPassBegin, &real)) {
      if (real == VK_NULL_HANDLE)
         return;
      begin.framebuffer = real;
   }

   wcb->device->dispatch_table.CmdBeginRenderPass(wcb->dispatch_handle, &begin,
                                                  contents);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBeginRenderPass2(VkCommandBuffer commandBuffer,
                            const VkRenderPassBeginInfo *pRenderPassBegin,
                            const VkSubpassBeginInfo *pSubpassBeginInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkRenderPassBeginInfo begin = *pRenderPassBegin;
   VkFramebuffer real;

   if (wrapper_lower_imageless_framebuffer(wcb, pRenderPassBegin, &real)) {
      if (real == VK_NULL_HANDLE)
         return;
      begin.framebuffer = real;
   }

   if (device->dispatch_table.CmdBeginRenderPass2)
      device->dispatch_table.CmdBeginRenderPass2(wcb->dispatch_handle, &begin,
                                                 pSubpassBeginInfo);
   else
      device->dispatch_table.CmdBeginRenderPass2KHR(wcb->dispatch_handle,
                                                    &begin, pSubpassBeginInfo);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyImage(VkDevice _device,
					 VkImage image,
					 const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   struct wrapper_image *wi = get_wrapper_image_from_handle(device, image);
   wrapper_bcn_hq_image_destroy(device, wi);
   wrapper_image_destroy(device, wi, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex,
                       uint32_t queueIndex, VkQueue* pQueue) {
   vk_common_GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceQueue2(VkDevice _device, const VkDeviceQueueInfo2* pQueueInfo,
                        VkQueue* pQueue) {
   VK_FROM_HANDLE(vk_device, device, _device);

   struct vk_queue *queue = NULL;
   vk_foreach_queue(iter, device) {
      if (iter->queue_family_index == pQueueInfo->queueFamilyIndex &&
          iter->index_in_family == pQueueInfo->queueIndex &&
          iter->flags == pQueueInfo->flags) {
         queue = iter;
         break;
      }
   }

   *pQueue = queue ? vk_queue_to_handle(queue) : VK_NULL_HANDLE;
}

/* ---- VK_KHR_maintenance5 emulation ------------------------------------------
 * DXVK > 2.4.1 requires VK_KHR_maintenance5, which Xclipse (and other drivers)
 * don't expose. We advertise + fake the feature (wrapper_physical_device.c) and
 * implement its new entrypoints here. Each prefers the base driver's native
 * implementation when present (so this is harmless on drivers that already have
 * maintenance5, e.g. Mali) and otherwise translates to the pre-maintenance5
 * equivalent. */

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer buffer,
                               VkDeviceSize offset, VkDeviceSize size,
                               VkIndexType indexType) {
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;

   if (buffer == VK_NULL_HANDLE && device->emulate_null_descriptor &&
       device->null_buffer != VK_NULL_HANDLE) {
      buffer = device->null_buffer;
      offset = 0;
      size = VK_WHOLE_SIZE;
   }

   if (device->dispatch_table.CmdBindIndexBuffer2)
      device->dispatch_table.CmdBindIndexBuffer2(wcb->dispatch_handle, buffer, offset, size, indexType);
   else if (device->dispatch_table.CmdBindIndexBuffer2KHR)
      device->dispatch_table.CmdBindIndexBuffer2KHR(wcb->dispatch_handle, buffer, offset, size, indexType);
   else /* pre-maintenance5: the size parameter is a robustness bound; dropping it is safe */
      device->dispatch_table.CmdBindIndexBuffer(wcb->dispatch_handle, buffer, offset, indexType);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetRenderingAreaGranularityKHR(VkDevice _device,
                                       const VkRenderingAreaInfo *pRenderingAreaInfo,
                                       VkExtent2D *pGranularity) {
   VK_FROM_HANDLE(wrapper_device, device, _device);

   if (device->dispatch_table.GetRenderingAreaGranularity)
      device->dispatch_table.GetRenderingAreaGranularity(device->dispatch_handle, pRenderingAreaInfo, pGranularity);
   else if (device->dispatch_table.GetRenderingAreaGranularityKHR)
      device->dispatch_table.GetRenderingAreaGranularityKHR(device->dispatch_handle, pRenderingAreaInfo, pGranularity);
   else /* {1,1} is always a valid render-area granularity */
      *pGranularity = (VkExtent2D){ 1, 1 };
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetImageSubresourceLayout2KHR(VkDevice _device, VkImage image,
                                      const VkImageSubresource2 *pSubresource,
                                      VkSubresourceLayout2 *pLayout) {
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkImageSubresource2 capped_sub;
   uint32_t mip_drop = wrapper_image_mip_drop(device, image);

   if (mip_drop) {
      capped_sub = *pSubresource;
      if (!wrapper_cap_level(mip_drop, &capped_sub.imageSubresource.mipLevel))
         capped_sub.imageSubresource.mipLevel = 0;
      pSubresource = &capped_sub;
   }

   if (device->dispatch_table.GetImageSubresourceLayout2)
      device->dispatch_table.GetImageSubresourceLayout2(device->dispatch_handle, image, pSubresource, pLayout);
   else if (device->dispatch_table.GetImageSubresourceLayout2KHR)
      device->dispatch_table.GetImageSubresourceLayout2KHR(device->dispatch_handle, image, pSubresource, pLayout);
   else if (device->dispatch_table.GetImageSubresourceLayout2EXT)
      device->dispatch_table.GetImageSubresourceLayout2EXT(device->dispatch_handle, image, pSubresource, pLayout);
   else
      device->dispatch_table.GetImageSubresourceLayout(device->dispatch_handle, image,
         &pSubresource->imageSubresource, &pLayout->subresourceLayout);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetDeviceImageSubresourceLayoutKHR(VkDevice _device,
                                           const VkDeviceImageSubresourceInfo *pInfo,
                                           VkSubresourceLayout2 *pLayout) {
   VK_FROM_HANDLE(wrapper_device, device, _device);

   if (device->dispatch_table.GetDeviceImageSubresourceLayout) {
      device->dispatch_table.GetDeviceImageSubresourceLayout(device->dispatch_handle, pInfo, pLayout);
      return;
   }
   if (device->dispatch_table.GetDeviceImageSubresourceLayoutKHR) {
      device->dispatch_table.GetDeviceImageSubresourceLayoutKHR(device->dispatch_handle, pInfo, pLayout);
      return;
   }

   /* Emulate via a transient image: create it, query the subresource layout,
    * destroy it. Use the base dispatch directly to avoid wrapper bookkeeping. */
   VkImage tmp;
   if (device->dispatch_table.CreateImage(device->dispatch_handle,
          pInfo->pCreateInfo, NULL, &tmp) == VK_SUCCESS) {
      device->dispatch_table.GetImageSubresourceLayout(device->dispatch_handle, tmp,
         &pInfo->pSubresource->imageSubresource, &pLayout->subresourceLayout);
      device->dispatch_table.DestroyImage(device->dispatch_handle, tmp, NULL);
   }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
wrapper_GetDeviceProcAddr(VkDevice _device, const char* pName) {
   VK_FROM_HANDLE(wrapper_device, device, _device);
   PFN_vkVoidFunction func = vk_device_get_proc_addr(&device->vk, pName);
   if (func || !pName)
      return func;

   static const char *const suffixes[] = { "KHR", "EXT" };
   for (unsigned i = 0; i < ARRAY_SIZE(suffixes); i++) {
      char alias[256];
      if (snprintf(alias, sizeof(alias), "%s%s", pName, suffixes[i]) >= (int)sizeof(alias))
         continue;
      func = vk_device_get_proc_addr(&device->vk, alias);
      if (func)
         return func;
   }
   return NULL;
}

/* ---- VK_KHR_push_descriptor emulation -------------------------------------
 * Drivers that lack push descriptors (e.g. Mali r44) can't run vkd3d/DXVK,
 * which require them. We translate each push into: allocate a descriptor set
 * from a per-command-buffer pool, update it, then bind it normally. Descriptor
 * set layouts get the push-bit stripped at creation (so they're allocatable);
 * push templates are converted to normal DESCRIPTOR_SET templates. Everything
 * here is a thin pass-through unless device->emulate_push_descriptor is set. */

#define WRAPPER_PUSH_POOL_CHUNK 64

static void
wrapper_push_pool_reset_all(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   for (struct wrapper_push_pool *p = wcb->push_pools; p; p = p->next) {
      device->dispatch_table.ResetDescriptorPool(device->dispatch_handle, p->pool, 0);
      p->remaining = WRAPPER_PUSH_POOL_CHUNK;
   }
}

static void
wrapper_push_pool_destroy_all(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_push_pool *p = wcb->push_pools;
   while (p) {
      struct wrapper_push_pool *next = p->next;
      device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, p->pool, NULL);
      free(p);
      p = next;
   }
   wcb->push_pools = NULL;
}

/* Allocate one descriptor set of `layout` from a per-CB chunked pool sized from
 * the layout's bindings. Pools specialize per layout: an allocation that can't
 * be served by an existing pool spins up a new one. */
static VkResult
wrapper_push_alloc_set(struct wrapper_command_buffer *wcb,
                       VkDescriptorSetLayout layout,
                       const struct wrapper_push_dsl *dsl,
                       VkDescriptorSet *out)
{
   struct wrapper_device *device = wcb->device;
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;

   struct wrapper_push_pool *p = wcb->push_pools;
   for (; p; p = p->next)
      if (p->layout == layout && p->remaining > 0)
         break;

   if (!p) {
      VkDescriptorPoolSize sizes[16];
      for (uint32_t i = 0; i < dsl->size_count; i++) {
         sizes[i].type = dsl->sizes[i].type;
         sizes[i].descriptorCount =
            dsl->sizes[i].descriptorCount * WRAPPER_PUSH_POOL_CHUNK;
      }
      VkDescriptorPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = WRAPPER_PUSH_POOL_CHUNK,
         .poolSizeCount = dsl->size_count,
         .pPoolSizes = sizes,
      };
      VkDescriptorPool pool;
      VkResult r = dt->CreateDescriptorPool(device->dispatch_handle, &pci, NULL, &pool);
      if (r != VK_SUCCESS)
         return r;
      p = malloc(sizeof(*p));
      if (!p) {
         dt->DestroyDescriptorPool(device->dispatch_handle, pool, NULL);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      p->pool = pool;
      p->layout = layout;
      p->remaining = WRAPPER_PUSH_POOL_CHUNK;
      p->next = wcb->push_pools;
      wcb->push_pools = p;
   }

   VkDescriptorSetAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = p->pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
   };
   VkResult r = dt->AllocateDescriptorSets(device->dispatch_handle, &ai, out);
   if (r == VK_SUCCESS)
      p->remaining--;
   return r;
}

/* Resolve (pipelineLayout, set) -> the push set's layout handle + its record. */
static struct wrapper_push_dsl *
wrapper_push_resolve(struct wrapper_device *device, VkPipelineLayout layout,
                     uint32_t set, VkDescriptorSetLayout *out_handle)
{
   struct wrapper_push_dsl *dsl = NULL;
   *out_handle = VK_NULL_HANDLE;
   simple_mtx_lock(&device->push_mutex);
   struct wrapper_push_pl *pl =
      _mesa_hash_table_u64_search(device->push_pl_table, (uint64_t)layout);
   if (pl && set < pl->set_layout_count) {
      *out_handle = pl->set_layouts[set];
      dsl = _mesa_hash_table_u64_search(device->push_dsl_table,
                                        (uint64_t)*out_handle);
   }
   simple_mtx_unlock(&device->push_mutex);
   return dsl;
}

static void
wrapper_emulate_push_descriptor(struct wrapper_command_buffer *wcb,
                                VkPipelineBindPoint bindPoint,
                                VkPipelineLayout layout, uint32_t set,
                                uint32_t writeCount,
                                const VkWriteDescriptorSet *pWrites)
{
   struct wrapper_device *device = wcb->device;
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;

   VkDescriptorSetLayout dsl_handle;
   struct wrapper_push_dsl *dsl =
      wrapper_push_resolve(device, layout, set, &dsl_handle);

   if (!dsl || dsl->size_count == 0 || dsl_handle == VK_NULL_HANDLE) {
      if (dt->CmdPushDescriptorSetKHR)
         dt->CmdPushDescriptorSetKHR(wcb->dispatch_handle, bindPoint, layout,
                                     set, writeCount, pWrites);
      else
         WRAPPER_LOG(error, "push descriptor: no layout record for set %u", set);
      return;
   }

   VkDescriptorSet dset;
   if (wrapper_push_alloc_set(wcb, dsl_handle, dsl, &dset) != VK_SUCCESS)
      return;

   /* Point the writes at our set and route through wrapper_UpdateDescriptorSets
    * so nullDescriptor emulation (if active) still applies. */
   VkWriteDescriptorSet *writes = malloc(sizeof(*writes) * writeCount);
   if (!writes)
      return;
   memcpy(writes, pWrites, sizeof(*writes) * writeCount);
   for (uint32_t i = 0; i < writeCount; i++)
      writes[i].dstSet = dset;
   wrapper_UpdateDescriptorSets(wrapper_device_to_handle(device),
                                writeCount, writes, 0, NULL);
   free(writes);

   dt->CmdBindDescriptorSets(wcb->dispatch_handle, bindPoint, layout,
                             device->fold_sets ? wrapper_fold_set_index(device, layout, set) : set,
                             1, &dset, 0, NULL);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer,
                               VkPipelineBindPoint pipelineBindPoint,
                               VkPipelineLayout layout, uint32_t set,
                               uint32_t descriptorWriteCount,
                               const VkWriteDescriptorSet *pDescriptorWrites)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (!wcb->device->emulate_push_descriptor) {
      wcb->device->dispatch_table.CmdPushDescriptorSetKHR(wcb->dispatch_handle,
         pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
      return;
   }
   wrapper_emulate_push_descriptor(wcb, pipelineBindPoint, layout, set,
                                   descriptorWriteCount, pDescriptorWrites);
}

static void
wrapper_emulate_push_descriptor_template(struct wrapper_command_buffer *wcb,
                                         VkDescriptorUpdateTemplate tpl_handle,
                                         VkPipelineLayout layout, uint32_t set,
                                         const void *pData)
{
   struct wrapper_device *device = wcb->device;
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;

   simple_mtx_lock(&device->push_mutex);
   struct wrapper_push_template *tpl =
      _mesa_hash_table_u64_search(device->push_template_table, (uint64_t)tpl_handle);
   VkPipelineBindPoint bp = tpl ? tpl->bind_point : VK_PIPELINE_BIND_POINT_GRAPHICS;
   simple_mtx_unlock(&device->push_mutex);

   VkDescriptorSetLayout dsl_handle;
   struct wrapper_push_dsl *dsl =
      wrapper_push_resolve(device, layout, set, &dsl_handle);

   if (!dsl || dsl->size_count == 0 || dsl_handle == VK_NULL_HANDLE) {
      if (dt->CmdPushDescriptorSetWithTemplateKHR)
         dt->CmdPushDescriptorSetWithTemplateKHR(wcb->dispatch_handle,
            tpl_handle, layout, set, pData);
      return;
   }

   VkDescriptorSet dset;
   if (wrapper_push_alloc_set(wcb, dsl_handle, dsl, &dset) != VK_SUCCESS)
      return;

   /* The template was rewritten to DESCRIPTOR_SET type at creation, so the
    * driver's normal template apply works on our allocated set. */
   PFN_vkUpdateDescriptorSetWithTemplate upd =
      dt->UpdateDescriptorSetWithTemplate ? dt->UpdateDescriptorSetWithTemplate
                                          : dt->UpdateDescriptorSetWithTemplateKHR;
   upd(device->dispatch_handle, dset, tpl_handle, pData);
   dt->CmdBindDescriptorSets(wcb->dispatch_handle, bp, layout,
                             device->fold_sets ? wrapper_fold_set_index(device, layout, set) : set,
                             1, &dset, 0, NULL);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer commandBuffer,
                                           VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                           VkPipelineLayout layout, uint32_t set,
                                           const void *pData)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (!wcb->device->emulate_push_descriptor) {
      wcb->device->dispatch_table.CmdPushDescriptorSetWithTemplateKHR(
         wcb->dispatch_handle, descriptorUpdateTemplate, layout, set, pData);
      return;
   }
   wrapper_emulate_push_descriptor_template(wcb, descriptorUpdateTemplate,
                                            layout, set, pData);
}

static VkResult
wrapper_create_descriptor_set_layout(VkDevice _device,
                                     const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                     const VkAllocationCallbacks *pAllocator,
                                     VkDescriptorSetLayout *pSetLayout);

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDescriptorSetLayout(VkDevice _device,
                                 const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                 const VkAllocationCallbacks *pAllocator,
                                 VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult r = wrapper_create_descriptor_set_layout(_device, pCreateInfo, pAllocator,
                                                     pSetLayout);
   if (r == VK_SUCCESS && device->fold_sets)
      wrapper_fold_note_layout(device, pCreateInfo, *pSetLayout);
   return r;
}

static VkResult
wrapper_create_descriptor_set_layout(VkDevice _device,
                                     const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                     const VkAllocationCallbacks *pAllocator,
                                     VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;

   if (!device->emulate_push_descriptor)
      return dt->CreateDescriptorSetLayout(device->dispatch_handle,
                                           pCreateInfo, pAllocator, pSetLayout);

   VkDescriptorSetLayoutCreateInfo ci = *pCreateInfo;
   bool is_push = (ci.flags &
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) != 0;
   if (is_push)
      ci.flags &= ~VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

   VkResult r = dt->CreateDescriptorSetLayout(device->dispatch_handle,
                                              &ci, pAllocator, pSetLayout);
   if (r != VK_SUCCESS)
      return r;

   struct wrapper_push_dsl *rec = calloc(1, sizeof(*rec));
   if (rec) {
      rec->is_push = is_push;
      for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
         const VkDescriptorSetLayoutBinding *b = &pCreateInfo->pBindings[i];
         if (b->descriptorCount == 0)
            continue;
         uint32_t j;
         for (j = 0; j < rec->size_count; j++)
            if (rec->sizes[j].type == b->descriptorType) {
               rec->sizes[j].descriptorCount += b->descriptorCount;
               break;
            }
         if (j == rec->size_count && rec->size_count < 16) {
            rec->sizes[j].type = b->descriptorType;
            rec->sizes[j].descriptorCount = b->descriptorCount;
            rec->size_count++;
         }
      }
      simple_mtx_lock(&device->push_mutex);
      _mesa_hash_table_u64_insert(device->push_dsl_table,
                                  (uint64_t)*pSetLayout, rec);
      simple_mtx_unlock(&device->push_mutex);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyDescriptorSetLayout(VkDevice _device,
                                  VkDescriptorSetLayout descriptorSetLayout,
                                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (device->emulate_push_descriptor && descriptorSetLayout) {
      simple_mtx_lock(&device->push_mutex);
      struct wrapper_push_dsl *rec = _mesa_hash_table_u64_search(
         device->push_dsl_table, (uint64_t)descriptorSetLayout);
      if (rec) {
         _mesa_hash_table_u64_remove(device->push_dsl_table,
                                     (uint64_t)descriptorSetLayout);
         free(rec);
      }
      simple_mtx_unlock(&device->push_mutex);
   }
   device->dispatch_table.DestroyDescriptorSetLayout(device->dispatch_handle,
      descriptorSetLayout, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                         VkShaderStageFlags stageFlags, uint32_t offset,
                         uint32_t size, const void *pValues)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   const struct wrapper_physical_device *pdev = wcb->device->physical;

   if (pdev->raises_push_constant_limit) {
      uint32_t limit = pdev->properties2.properties.limits.maxPushConstantsSize;
      if (offset >= limit) {
         WRAPPER_LOG(error, "Dropping push constants past the driver's %u bytes (offset %u size %u)",
                     limit, offset, size);
         return;
      }
      if (offset + size > limit) {
         WRAPPER_LOG(error, "Dropping push constants past the driver's %u bytes (offset %u size %u)",
                     limit, offset, size);
         size = limit - offset;
      }
   }
   if (wcb->device->bcn_bda)
      wrapper_cmd_track_push(wcb, layout, stageFlags, offset, size, pValues);
   wcb->device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, layout,
                                                stageFlags, offset, size, pValues);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreatePipelineLayout(VkDevice _device,
                            const VkPipelineLayoutCreateInfo *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult r;
   if (!device->fold_sets ||
       !wrapper_fold_create_pipeline_layout(device, pCreateInfo, pAllocator,
                                            pPipelineLayout, &r))
      r = device->dispatch_table.CreatePipelineLayout(
         device->dispatch_handle, pCreateInfo, pAllocator, pPipelineLayout);
   if (r != VK_SUCCESS || !device->emulate_push_descriptor)
      return r;

   struct wrapper_push_pl *rec = calloc(1, sizeof(*rec));
   if (rec) {
      rec->set_layout_count = pCreateInfo->setLayoutCount;
      if (rec->set_layout_count) {
         rec->set_layouts = malloc(sizeof(VkDescriptorSetLayout) *
                                   rec->set_layout_count);
         if (rec->set_layouts)
            memcpy(rec->set_layouts, pCreateInfo->pSetLayouts,
                   sizeof(VkDescriptorSetLayout) * rec->set_layout_count);
      }
      simple_mtx_lock(&device->push_mutex);
      _mesa_hash_table_u64_insert(device->push_pl_table,
                                  (uint64_t)*pPipelineLayout, rec);
      simple_mtx_unlock(&device->push_mutex);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyPipelineLayout(VkDevice _device, VkPipelineLayout pipelineLayout,
                             const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (device->emulate_push_descriptor && pipelineLayout) {
      simple_mtx_lock(&device->push_mutex);
      struct wrapper_push_pl *rec = _mesa_hash_table_u64_search(
         device->push_pl_table, (uint64_t)pipelineLayout);
      if (rec) {
         _mesa_hash_table_u64_remove(device->push_pl_table,
                                     (uint64_t)pipelineLayout);
         free(rec->set_layouts);
         free(rec);
      }
      simple_mtx_unlock(&device->push_mutex);
   }
   if (device->fold_sets && pipelineLayout)
      wrapper_fold_destroy_pipeline_layout(device, pipelineLayout);
   device->dispatch_table.DestroyPipelineLayout(device->dispatch_handle,
      pipelineLayout, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDescriptorPool(VkDevice _device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator,
                             VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (device->fold_sets)
      wrapper_fold_note_pool(device, pCreateInfo);
   return device->dispatch_table.CreateDescriptorPool(device->dispatch_handle, pCreateInfo,
                                                      pAllocator, pDescriptorPool);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool descriptorPool,
                              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (device->fold_sets && descriptorPool)
      wrapper_fold_reset_pool(device, descriptorPool);
   device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, descriptorPool,
                                                pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_ResetDescriptorPool(VkDevice _device, VkDescriptorPool descriptorPool,
                            VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (device->fold_sets)
      wrapper_fold_reset_pool(device, descriptorPool);
   return device->dispatch_table.ResetDescriptorPool(device->dispatch_handle, descriptorPool,
                                                     flags);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateDescriptorSets(VkDevice _device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                               VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult r = device->dispatch_table.AllocateDescriptorSets(device->dispatch_handle,
                                                              pAllocateInfo, pDescriptorSets);
   if (r == VK_SUCCESS && device->fold_sets)
      wrapper_fold_note_sets(device, pAllocateInfo, pDescriptorSets);
   return r;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_FreeDescriptorSets(VkDevice _device, VkDescriptorPool descriptorPool,
                           uint32_t descriptorSetCount, const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (device->fold_sets)
      wrapper_fold_free_sets(device, descriptorSetCount, pDescriptorSets);
   return device->dispatch_table.FreeDescriptorSets(device->dispatch_handle, descriptorPool,
                                                    descriptorSetCount, pDescriptorSets);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                              VkPipelineLayout layout, uint32_t firstSet,
                              uint32_t descriptorSetCount, const VkDescriptorSet *pDescriptorSets,
                              uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->fold_sets &&
       wrapper_fold_bind(wcb, pipelineBindPoint, layout, firstSet, descriptorSetCount,
                         pDescriptorSets, dynamicOffsetCount, pDynamicOffsets))
      return;
   wcb->device->dispatch_table.CmdBindDescriptorSets(wcb->dispatch_handle, pipelineBindPoint,
      layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount,
      pDynamicOffsets);
}

static VkResult
wrapper_create_update_template(struct wrapper_device *device,
   const VkDescriptorUpdateTemplateCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkDescriptorUpdateTemplate *pTemplate)
{
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   PFN_vkCreateDescriptorUpdateTemplate create_fn =
      dt->CreateDescriptorUpdateTemplate ? dt->CreateDescriptorUpdateTemplate
                                         : dt->CreateDescriptorUpdateTemplateKHR;

   if (!device->emulate_push_descriptor)
      return create_fn(device->dispatch_handle, pCreateInfo, pAllocator, pTemplate);

   VkDescriptorUpdateTemplateCreateInfo ci = *pCreateInfo;
   bool is_push =
      ci.templateType == VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS;
   if (is_push) {
      ci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
      /* A DESCRIPTOR_SET template needs a concrete set layout: the push set's. */
      VkDescriptorSetLayout dsl_handle;
      wrapper_push_resolve(device, ci.pipelineLayout, ci.set, &dsl_handle);
      ci.descriptorSetLayout = dsl_handle;
   }

   VkResult r = create_fn(device->dispatch_handle, &ci, pAllocator, pTemplate);
   if (r != VK_SUCCESS)
      return r;

   struct wrapper_push_template *rec = calloc(1, sizeof(*rec));
   if (rec) {
      rec->is_push = is_push;
      rec->bind_point = pCreateInfo->pipelineBindPoint;
      rec->pipeline_layout = pCreateInfo->pipelineLayout;
      rec->set = pCreateInfo->set;
      simple_mtx_lock(&device->push_mutex);
      _mesa_hash_table_u64_insert(device->push_template_table,
                                  (uint64_t)*pTemplate, rec);
      simple_mtx_unlock(&device->push_mutex);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDescriptorUpdateTemplate(VkDevice _device,
   const VkDescriptorUpdateTemplateCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkDescriptorUpdateTemplate *pDescriptorUpdateTemplate)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   return wrapper_create_update_template(device, pCreateInfo, pAllocator,
                                         pDescriptorUpdateTemplate);
}

static void
wrapper_destroy_update_template(struct wrapper_device *device,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate,
   const VkAllocationCallbacks *pAllocator)
{
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   if (device->emulate_push_descriptor && descriptorUpdateTemplate) {
      simple_mtx_lock(&device->push_mutex);
      struct wrapper_push_template *rec = _mesa_hash_table_u64_search(
         device->push_template_table, (uint64_t)descriptorUpdateTemplate);
      if (rec) {
         _mesa_hash_table_u64_remove(device->push_template_table,
                                     (uint64_t)descriptorUpdateTemplate);
         free(rec);
      }
      simple_mtx_unlock(&device->push_mutex);
   }
   PFN_vkDestroyDescriptorUpdateTemplate destroy_fn =
      dt->DestroyDescriptorUpdateTemplate ? dt->DestroyDescriptorUpdateTemplate
                                          : dt->DestroyDescriptorUpdateTemplateKHR;
   destroy_fn(device->dispatch_handle, descriptorUpdateTemplate, pAllocator);
}

static void
wrapper_dynamic_render_objects_reset(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   list_for_each_entry_safe(struct wrapper_dynamic_render_object, object,
                            &wcb->dynamic_render_objects, link) {
      if (object->framebuffer)
         device->dispatch_table.DestroyFramebuffer(device->dispatch_handle,
                                                    object->framebuffer, NULL);
      if (object->render_pass)
         device->dispatch_table.DestroyRenderPass(device->dispatch_handle,
                                                   object->render_pass, NULL);
      list_del(&object->link);
      free(object);
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyDescriptorUpdateTemplate(VkDevice _device,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   wrapper_destroy_update_template(device, descriptorUpdateTemplate, pAllocator);
}

/* Buffers the BCn emulation records into a command buffer. They are freed when
 * the command buffer is begun, reset or freed again, when it cannot be pending. */
static void
wrapper_bcn_transients_free(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   wrapper_bcn_hq_cmd_reset(wcb);
   list_for_each_entry_safe(struct wrapper_buffer, wb, &wcb->bcn_transients, link) {
      if (wb->dispatch_handle)
         device->dispatch_table.DestroyBuffer(device->dispatch_handle,
            wb->dispatch_handle, NULL);
      if (wb->bcn_image)
         device->dispatch_table.DestroyImage(device->dispatch_handle, wb->bcn_image, NULL);
      if (wb->memory)
         device->dispatch_table.FreeMemory(device->dispatch_handle, wb->memory, NULL);
      list_del(&wb->link);
      vk_object_free(&device->vk, &device->vk.alloc, wb);
   }
}

static void
wrapper_bcn_hq_barriers(struct wrapper_command_buffer *wcb, uint32_t count,
                        const VkImageMemoryBarrier *barriers)
{
   if (!__atomic_load_n(&wcb->device->bcn_hq_active, __ATOMIC_RELAXED))
      return;
   for (uint32_t i = 0; i < count; i++)
      wrapper_bcn_hq_barrier(wcb, barriers[i].image, barriers[i].oldLayout,
                             barriers[i].newLayout, barriers[i].srcQueueFamilyIndex,
                             barriers[i].dstQueueFamilyIndex);
}

static void
wrapper_bcn_hq_barriers2(struct wrapper_command_buffer *wcb, const VkDependencyInfo *dep)
{
   if (!__atomic_load_n(&wcb->device->bcn_hq_active, __ATOMIC_RELAXED))
      return;
   for (uint32_t i = 0; i < dep->imageMemoryBarrierCount; i++) {
      const VkImageMemoryBarrier2 *b = &dep->pImageMemoryBarriers[i];
      wrapper_bcn_hq_barrier(wcb, b->image, b->oldLayout, b->newLayout,
                             b->srcQueueFamilyIndex, b->dstQueueFamilyIndex);
   }
}
static VkResult
wrapper_create_legacy_render_pass_for_pipeline(
   struct wrapper_device *device,
   const VkPipelineRenderingCreateInfo *rendering,
   VkSampleCountFlagBits samples,
   VkRenderPass *render_pass);

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                          const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wrapper_dynamic_render_objects_reset(wcb);
   wrapper_stride_reset(wcb);
   wrapper_bcn_transients_free(wcb);
   wrapper_cmd_track_reset(wcb);
   wrapper_divisor_reset(wcb);
   if (wcb->device->emulate_push_descriptor)
      wrapper_push_pool_reset_all(wcb);

   VkCommandBufferBeginInfo legacy_begin;
   VkCommandBufferInheritanceInfo legacy_inheritance;
   const VkCommandBufferInheritanceRenderingInfo *rendering =
      pBeginInfo->pInheritanceInfo
         ? vk_find_struct_const(pBeginInfo->pInheritanceInfo->pNext,
                                COMMAND_BUFFER_INHERITANCE_RENDERING_INFO)
         : NULL;
   if (rendering && wrapper_lowers_core13(wcb->device) &&
       !wcb->device->dispatch_table.CmdBeginRendering &&
       pBeginInfo->pInheritanceInfo->renderPass == VK_NULL_HANDLE &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      VkPipelineRenderingCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
         .viewMask = rendering->viewMask,
         .colorAttachmentCount = rendering->colorAttachmentCount,
         .pColorAttachmentFormats = rendering->pColorAttachmentFormats,
         .depthAttachmentFormat = rendering->depthAttachmentFormat,
         .stencilAttachmentFormat = rendering->stencilAttachmentFormat,
      };
      VkRenderPass render_pass = VK_NULL_HANDLE;
      struct wrapper_dynamic_render_object *object = calloc(1, sizeof(*object));
      if (object &&
          wrapper_create_legacy_render_pass_for_pipeline(
             wcb->device, &info,
             rendering->rasterizationSamples ? rendering->rasterizationSamples
                                             : VK_SAMPLE_COUNT_1_BIT,
             &render_pass) == VK_SUCCESS) {
         object->render_pass = render_pass;
         list_addtail(&object->link, &wcb->dynamic_render_objects);
         legacy_inheritance = *pBeginInfo->pInheritanceInfo;
         legacy_inheritance.pNext = NULL;
         legacy_inheritance.renderPass = render_pass;
         legacy_inheritance.subpass = 0;
         legacy_inheritance.framebuffer = VK_NULL_HANDLE;
         legacy_begin = *pBeginInfo;
         legacy_begin.pInheritanceInfo = &legacy_inheritance;
         pBeginInfo = &legacy_begin;
      } else {
         free(object);
         WRAPPER_LOG(error, "Failed to lower dynamic-rendering inheritance");
      }
   }

   VkResult begin_result =
      wcb->device->dispatch_table.BeginCommandBuffer(wcb->dispatch_handle,
                                                     pBeginInfo);
   if (begin_result == VK_SUCCESS && wcb->device->cmdbuf_entry_barrier &&
       !wcb->secondary) {
      VkMemoryBarrier barrier = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      };
      wcb->device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         0, 1, &barrier, 0, NULL, 0, NULL);
   }
   return begin_result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   VkResult result = wcb->device->dispatch_table.EndCommandBuffer(
      wcb->dispatch_handle);
   return result;
}

static VkPipelineStageFlags
wrapper_stage_mask2_to_legacy(VkPipelineStageFlags2 stages, bool source)
{
   VkPipelineStageFlags legacy = (VkPipelineStageFlags)stages;

   /* Synchronization2 added stage bits above the legacy 32-bit range.  The
    * Tegra ICD cannot name those stages individually, so widen the dependency
    * to all commands.  This is conservative, but preserves ordering. */
   if (stages >> 32)
      legacy |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

   if (!legacy)
      legacy = source ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                      : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

   return legacy;
}

static VkAccessFlags
wrapper_access_mask2_to_legacy(VkAccessFlags2 access)
{
   VkAccessFlags legacy = (VkAccessFlags)access;

   /* Access2 also has bits with no legacy spelling.  MEMORY_READ/WRITE are
    * valid conservative substitutes when paired with ALL_COMMANDS. */
   if (access >> 32)
      legacy |= VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

   return legacy;
}

static void
wrapper_pipeline_barrier2(struct wrapper_command_buffer *wcb,
                          const VkDependencyInfo *pDependencyInfo)
{
   struct wrapper_device *device = wcb->device;

   if (device->dispatch_table.CmdPipelineBarrier2) {
      device->dispatch_table.CmdPipelineBarrier2(wcb->dispatch_handle,
                                                  pDependencyInfo);
      return;
   }
   if (!wrapper_lowers_core13(device))
      return;

   const uint32_t memory_count = pDependencyInfo->memoryBarrierCount;
   const uint32_t buffer_count = pDependencyInfo->bufferMemoryBarrierCount;
   const uint32_t image_count = pDependencyInfo->imageMemoryBarrierCount;
   VkMemoryBarrier *memory = calloc(memory_count, sizeof(*memory));
   VkBufferMemoryBarrier *buffers = calloc(buffer_count, sizeof(*buffers));
   VkImageMemoryBarrier *images = calloc(image_count, sizeof(*images));
   VkPipelineStageFlags src_stages = 0;
   VkPipelineStageFlags dst_stages = 0;

   if ((!memory && memory_count) || (!buffers && buffer_count) ||
       (!images && image_count)) {
      /* Command recording entry points cannot report allocation failures.
       * Preserve safety by recording a full memory dependency instead. */
      VkMemoryBarrier fallback = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                          VK_ACCESS_MEMORY_WRITE_BIT,
      };
      device->dispatch_table.CmdPipelineBarrier(
         wcb->dispatch_handle, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, pDependencyInfo->dependencyFlags,
         1, &fallback, 0, NULL, 0, NULL);
      free(memory);
      free(buffers);
      free(images);
      return;
   }

   for (uint32_t i = 0; i < memory_count; i++) {
      const VkMemoryBarrier2 *src = &pDependencyInfo->pMemoryBarriers[i];
      src_stages |= wrapper_stage_mask2_to_legacy(src->srcStageMask, true);
      dst_stages |= wrapper_stage_mask2_to_legacy(src->dstStageMask, false);
      memory[i] = (VkMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .pNext = src->pNext,
         .srcAccessMask = wrapper_access_mask2_to_legacy(src->srcAccessMask),
         .dstAccessMask = wrapper_access_mask2_to_legacy(src->dstAccessMask),
      };
   }

   for (uint32_t i = 0; i < buffer_count; i++) {
      const VkBufferMemoryBarrier2 *src =
         &pDependencyInfo->pBufferMemoryBarriers[i];
      src_stages |= wrapper_stage_mask2_to_legacy(src->srcStageMask, true);
      dst_stages |= wrapper_stage_mask2_to_legacy(src->dstStageMask, false);
      buffers[i] = (VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .pNext = src->pNext,
         .srcAccessMask = wrapper_access_mask2_to_legacy(src->srcAccessMask),
         .dstAccessMask = wrapper_access_mask2_to_legacy(src->dstAccessMask),
         .srcQueueFamilyIndex = src->srcQueueFamilyIndex,
         .dstQueueFamilyIndex = src->dstQueueFamilyIndex,
         .buffer = src->buffer,
         .offset = src->offset,
         .size = src->size,
      };
   }

   for (uint32_t i = 0; i < image_count; i++) {
      const VkImageMemoryBarrier2 *src =
         &pDependencyInfo->pImageMemoryBarriers[i];
      src_stages |= wrapper_stage_mask2_to_legacy(src->srcStageMask, true);
      dst_stages |= wrapper_stage_mask2_to_legacy(src->dstStageMask, false);
      images[i] = (VkImageMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .pNext = src->pNext,
         .srcAccessMask = wrapper_access_mask2_to_legacy(src->srcAccessMask),
         .dstAccessMask = wrapper_access_mask2_to_legacy(src->dstAccessMask),
         .oldLayout = src->oldLayout,
         .newLayout = src->newLayout,
         .srcQueueFamilyIndex = src->srcQueueFamilyIndex,
         .dstQueueFamilyIndex = src->dstQueueFamilyIndex,
         .image = src->image,
         .subresourceRange = src->subresourceRange,
      };
   }

   if (!src_stages)
      src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
   if (!dst_stages)
      dst_stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

   device->dispatch_table.CmdPipelineBarrier(
      wcb->dispatch_handle, src_stages, dst_stages,
      pDependencyInfo->dependencyFlags, memory_count, memory, buffer_count,
      buffers, image_count, images);

   free(memory);
   free(buffers);
   free(images);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                            const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   uint32_t count = pDependencyInfo->imageMemoryBarrierCount;
   VkImageMemoryBarrier2 stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageMemoryBarrier2 *capped = NULL;
   VkDependencyInfo capped_dep;

   if (count && __atomic_load_n(&wrapper_capped_images, __ATOMIC_RELAXED) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), count))) {
      capped_dep = *pDependencyInfo;
      capped_dep.imageMemoryBarrierCount = wrapper_cap_image_barriers2(wcb->device,
         pDependencyInfo->pImageMemoryBarriers, count, capped);
      capped_dep.pImageMemoryBarriers = capped;
      pDependencyInfo = &capped_dep;
   }

   wrapper_pipeline_barrier2(wcb, pDependencyInfo);
   wrapper_bcn_hq_barriers2(wcb, pDependencyInfo);
   wrapper_cap_free(capped, stack);
}

#define WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS 8
#define WRAPPER_DYNAMIC_MAX_ATTACHMENTS \
   (WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS * 2 + 2)

static const VkPipelineRenderingCreateInfo *
wrapper_find_pipeline_rendering_info(const void *pNext)
{
   const VkBaseInStructure *current = pNext;
   while (current) {
      if (current->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
         return (const VkPipelineRenderingCreateInfo *)current;
      current = current->pNext;
   }
   return NULL;
}

static const void *
wrapper_chain_without(void **mem, const void *pNext, VkStructureType skip)
{
   if (!vk_find_struct_const(pNext, PIPELINE_RENDERING_CREATE_INFO))
      return pNext;
   if (!*mem)
      *mem = ralloc_context(NULL);
   VkBaseOutStructure *head = NULL, *tail = NULL;
   for (const VkBaseInStructure *in = pNext; in; in = in->pNext) {
      if (in->sType == skip)
         continue;
      VkBaseOutStructure *out = ralloc_memdup(*mem, in, vk_structure_type_size(in));
      out->pNext = NULL;
      if (tail)
         tail->pNext = out;
      else
         head = out;
      tail = out;
   }
   return head;
}

static VkResult
wrapper_create_legacy_render_pass_for_pipeline(
   struct wrapper_device *device,
   const VkPipelineRenderingCreateInfo *rendering,
   VkSampleCountFlagBits samples,
   VkRenderPass *render_pass)
{
   if (rendering->colorAttachmentCount >
       WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   VkAttachmentDescription attachments[WRAPPER_DYNAMIC_MAX_ATTACHMENTS] = {0};
   VkAttachmentReference colors[WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS];
   uint32_t attachment_count = 0;

   for (uint32_t i = 0; i < rendering->colorAttachmentCount; i++) {
      colors[i] = (VkAttachmentReference) {
         .attachment = VK_ATTACHMENT_UNUSED,
         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
      if (rendering->pColorAttachmentFormats[i] == VK_FORMAT_UNDEFINED)
         continue;
      colors[i].attachment = attachment_count;
      attachments[attachment_count++] = (VkAttachmentDescription) {
         .format = rendering->pColorAttachmentFormats[i],
         .samples = samples,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
   }

   VkAttachmentReference depth = {
      .attachment = VK_ATTACHMENT_UNUSED,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   VkFormat depth_format = rendering->depthAttachmentFormat != VK_FORMAT_UNDEFINED
      ? rendering->depthAttachmentFormat : rendering->stencilAttachmentFormat;
   if (rendering->depthAttachmentFormat != VK_FORMAT_UNDEFINED &&
       rendering->stencilAttachmentFormat != VK_FORMAT_UNDEFINED &&
       rendering->depthAttachmentFormat != rendering->stencilAttachmentFormat) {
      WRAPPER_LOG(error,
         "Cannot lower dynamic rendering with different depth/stencil formats");
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   if (depth_format != VK_FORMAT_UNDEFINED) {
      depth.attachment = attachment_count;
      attachments[attachment_count++] = (VkAttachmentDescription) {
         .format = depth_format,
         .samples = samples,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
         .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      };
   }

   VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = rendering->colorAttachmentCount,
      .pColorAttachments = colors,
      .pDepthStencilAttachment = depth.attachment != VK_ATTACHMENT_UNUSED
         ? &depth : NULL,
   };
   VkRenderPassMultiviewCreateInfo multiview = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
      .subpassCount = 1,
      .pViewMasks = &rendering->viewMask,
   };
   VkRenderPassCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .pNext = rendering->viewMask ? &multiview : NULL,
      .attachmentCount = attachment_count,
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   return device->dispatch_table.CreateRenderPass(device->dispatch_handle,
                                                   &info, NULL, render_pass);
}

static bool
wrapper_lookup_image_view(struct wrapper_device *device, VkImageView handle,
                          struct wrapper_image_view *out)
{
   bool found = false;
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_image_view *view =
      _mesa_hash_table_u64_search(device->image_view_table, (uint64_t)handle);
   if (view) {
      *out = *view;
      found = true;
   }
   simple_mtx_unlock(&device->resource_mutex);
   return found;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBeginRendering(VkCommandBuffer commandBuffer,
                          const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;


   if (device->dispatch_table.CmdBeginRendering) {
      device->dispatch_table.CmdBeginRendering(wcb->dispatch_handle,
                                                pRenderingInfo);
      wcb->dynamic_rendering_active = true;
      return;
   }
   if (!wrapper_lowers_core13(device))
      return;

   if (pRenderingInfo->colorAttachmentCount >
       WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS) {
      WRAPPER_LOG(error, "Too many dynamic-rendering color attachments: %u",
                  pRenderingInfo->colorAttachmentCount);
      return;
   }

   VkAttachmentDescription attachments[WRAPPER_DYNAMIC_MAX_ATTACHMENTS] = {0};
   VkAttachmentReference colors[WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS];
   VkAttachmentReference resolves[WRAPPER_DYNAMIC_MAX_COLOR_ATTACHMENTS];
   VkImageView views[WRAPPER_DYNAMIC_MAX_ATTACHMENTS] = {0};
   VkClearValue clears[WRAPPER_DYNAMIC_MAX_ATTACHMENTS] = {0};
   uint32_t attachment_count = 0;
   bool has_resolve = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *src =
         &pRenderingInfo->pColorAttachments[i];
      colors[i] = (VkAttachmentReference) {
         .attachment = VK_ATTACHMENT_UNUSED,
         .layout = src->imageLayout,
      };
      resolves[i] = (VkAttachmentReference) {
         .attachment = VK_ATTACHMENT_UNUSED,
         .layout = src->resolveImageLayout,
      };
      if (src->imageView == VK_NULL_HANDLE)
         continue;

      struct wrapper_image_view view;
      if (!wrapper_lookup_image_view(device, src->imageView, &view)) {
         WRAPPER_LOG(error, "Dynamic rendering references unknown image view %p",
                     (void *)(uintptr_t)src->imageView);
         return;
      }
      colors[i].attachment = attachment_count;
      views[attachment_count] = src->imageView;
      clears[attachment_count] = src->clearValue;
      attachments[attachment_count++] = (VkAttachmentDescription) {
         .format = view.format,
         .samples = view.samples,
         .loadOp = src->loadOp,
         .storeOp = src->storeOp,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = src->imageLayout,
         .finalLayout = src->imageLayout,
      };

      if (src->resolveMode != VK_RESOLVE_MODE_NONE &&
          src->resolveImageView != VK_NULL_HANDLE) {
         struct wrapper_image_view resolve_view;
         if (!wrapper_lookup_image_view(device, src->resolveImageView,
                                        &resolve_view)) {
            WRAPPER_LOG(error, "Dynamic rendering references unknown resolve view");
            return;
         }
         has_resolve = true;
         resolves[i].attachment = attachment_count;
         views[attachment_count] = src->resolveImageView;
         attachments[attachment_count++] = (VkAttachmentDescription) {
            .format = resolve_view.format,
            .samples = resolve_view.samples,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = src->resolveImageLayout,
            .finalLayout = src->resolveImageLayout,
         };
      }
   }

   VkAttachmentReference depth_ref = {
      .attachment = VK_ATTACHMENT_UNUSED,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   const VkRenderingAttachmentInfo *depth_src = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *stencil_src = pRenderingInfo->pStencilAttachment;
   if (depth_src && stencil_src && depth_src->imageView != VK_NULL_HANDLE &&
       stencil_src->imageView != VK_NULL_HANDLE &&
       depth_src->imageView != stencil_src->imageView) {
      WRAPPER_LOG(error,
         "Cannot lower dynamic rendering with separate depth/stencil views");
      return;
   }

   const VkRenderingAttachmentInfo *ds_src =
      depth_src && depth_src->imageView != VK_NULL_HANDLE ? depth_src :
      stencil_src && stencil_src->imageView != VK_NULL_HANDLE ? stencil_src : NULL;
   if (ds_src) {
      struct wrapper_image_view view;
      if (!wrapper_lookup_image_view(device, ds_src->imageView, &view)) {
         WRAPPER_LOG(error, "Dynamic rendering references unknown depth view");
         return;
      }
      depth_ref.attachment = attachment_count;
      depth_ref.layout = ds_src->imageLayout;
      views[attachment_count] = ds_src->imageView;
      clears[attachment_count] = ds_src->clearValue;
      if (depth_src)
         clears[attachment_count].depthStencil.depth =
            depth_src->clearValue.depthStencil.depth;
      if (stencil_src)
         clears[attachment_count].depthStencil.stencil =
            stencil_src->clearValue.depthStencil.stencil;
      attachments[attachment_count++] = (VkAttachmentDescription) {
         .format = view.format,
         .samples = view.samples,
         .loadOp = depth_src ? depth_src->loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = depth_src ? depth_src->storeOp : VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .stencilLoadOp = stencil_src ? stencil_src->loadOp
                                      : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = stencil_src ? stencil_src->storeOp
                                       : VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = ds_src->imageLayout,
         .finalLayout = ds_src->imageLayout,
      };
      if ((depth_src && depth_src->resolveMode != VK_RESOLVE_MODE_NONE) ||
          (stencil_src && stencil_src->resolveMode != VK_RESOLVE_MODE_NONE))
         WRAPPER_LOG(error,
            "Depth/stencil dynamic-rendering resolve is not emulated");
   }

   VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
      .pColorAttachments = colors,
      .pResolveAttachments = has_resolve ? resolves : NULL,
      .pDepthStencilAttachment = depth_ref.attachment != VK_ATTACHMENT_UNUSED
         ? &depth_ref : NULL,
   };
   VkRenderPassMultiviewCreateInfo multiview = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
      .subpassCount = 1,
      .pViewMasks = &pRenderingInfo->viewMask,
   };
   VkRenderPassCreateInfo rp_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .pNext = pRenderingInfo->viewMask ? &multiview : NULL,
      .attachmentCount = attachment_count,
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass render_pass = VK_NULL_HANDLE;
   VkResult result = device->dispatch_table.CreateRenderPass(
      device->dispatch_handle, &rp_info, NULL, &render_pass);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to lower dynamic render pass: %d", result);
      return;
   }

   VkFramebufferCreateInfo fb_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = attachment_count,
      .pAttachments = views,
      .width = pRenderingInfo->renderArea.offset.x +
               pRenderingInfo->renderArea.extent.width,
      .height = pRenderingInfo->renderArea.offset.y +
                pRenderingInfo->renderArea.extent.height,
      .layers = pRenderingInfo->viewMask ? 1 : MAX2(1, pRenderingInfo->layerCount),
   };
   VkFramebuffer framebuffer = VK_NULL_HANDLE;
   result = device->dispatch_table.CreateFramebuffer(device->dispatch_handle,
                                                       &fb_info, NULL,
                                                       &framebuffer);
   if (result != VK_SUCCESS) {
      device->dispatch_table.DestroyRenderPass(device->dispatch_handle,
                                                render_pass, NULL);
      WRAPPER_LOG(error, "Failed to lower dynamic framebuffer: %d", result);
      return;
   }

   struct wrapper_dynamic_render_object *object = calloc(1, sizeof(*object));
   if (!object) {
      device->dispatch_table.DestroyFramebuffer(device->dispatch_handle,
                                                  framebuffer, NULL);
      device->dispatch_table.DestroyRenderPass(device->dispatch_handle,
                                                render_pass, NULL);
      return;
   }
   object->render_pass = render_pass;
   object->framebuffer = framebuffer;
   list_addtail(&object->link, &wcb->dynamic_render_objects);

   VkRenderPassBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = pRenderingInfo->renderArea,
      .clearValueCount = attachment_count,
      .pClearValues = clears,
   };
   device->dispatch_table.CmdBeginRenderPass(
      wcb->dispatch_handle, &begin,
      (pRenderingInfo->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)
         ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
         : VK_SUBPASS_CONTENTS_INLINE);
   wcb->dynamic_rendering_active = true;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->dispatch_table.CmdEndRendering)
      wcb->device->dispatch_table.CmdEndRendering(wcb->dispatch_handle);
   else if (wcb->dynamic_rendering_active)
      wcb->device->dispatch_table.CmdEndRenderPass(wcb->dispatch_handle);
   wcb->dynamic_rendering_active = false;
}

static bool
wrapper_lowers_maintenance5(const struct wrapper_device *device)
{
   return device->vk.enabled_extensions.KHR_maintenance5 &&
          !device->physical->base_supported_extensions.KHR_maintenance5;
}

static bool
wrapper_patches_spirv(const struct wrapper_device *device)
{
   return device->physical->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY;
}

static bool
wrapper_tracks_shader_code(const struct wrapper_device *device)
{
   return device->emulate_dynamic_stride || device->emulate_vertex_divisor ||
          device->emulate_eds || device->emulate_eds2 || device->fold_sets;
}

static bool
wrapper_lowers_inline_shaders(const struct wrapper_device *device)
{
   return wrapper_lowers_maintenance5(device) || wrapper_patches_spirv(device) ||
          wrapper_tracks_shader_code(device) || wrapper_lowers_core13(device);
}

static void
wrapper_scan_inline_shader(const VkPipelineShaderStageCreateInfo *stage)
{
   const VkShaderModuleCreateInfo *module_info;

   if (stage->module == VK_NULL_HANDLE &&
       (module_info = vk_find_struct_const(stage->pNext, SHADER_MODULE_CREATE_INFO)))
      bcn_scan_shader(module_info->pCode, module_info->codeSize);
}

static VkPipelineCreateFlags
wrapper_fold_pipeline_flags2(const void **pNext, VkPipelineCreateFlags flags)
{
   const VkPipelineCreateFlags2CreateInfo *flags2 =
      vk_find_struct_const(*pNext, PIPELINE_CREATE_FLAGS_2_CREATE_INFO);
   if (!flags2)
      return flags;
   if (*pNext == (const void *)flags2)
      *pNext = flags2->pNext;
   return (VkPipelineCreateFlags)flags2->flags;
}

static VkResult
wrapper_lower_inline_shader_stage(struct wrapper_device *device,
                                  VkPipelineShaderStageCreateInfo *stage,
                                  VkShaderModule *modules, uint32_t *module_count)
{
   const VkShaderModuleCreateInfo *module_info;

   if (stage->module != VK_NULL_HANDLE ||
       !(module_info = vk_find_struct_const(stage->pNext, SHADER_MODULE_CREATE_INFO)))
      return VK_SUCCESS;

   VkShaderModuleCreateInfo info = *module_info;
   info.pNext = NULL;
   VkResult result = wrapper_CreateShaderModule(wrapper_device_to_handle(device),
                                                &info, NULL, &modules[*module_count]);
   if (result != VK_SUCCESS)
      return result;

   stage->module = modules[(*module_count)++];
   if (stage->pNext == (const void *)module_info)
      stage->pNext = module_info->pNext;
   return VK_SUCCESS;
}

static void
wrapper_destroy_lowered_modules(struct wrapper_device *device,
                                VkShaderModule *modules, uint32_t count)
{
   for (uint32_t i = 0; i < count; i++) {
      if (wrapper_tracks_shader_code(device))
         wrapper_stride_forget_module(device, modules[i]);
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle,
                                                 modules[i], NULL);
   }
   free(modules);
}

static VkResult
wrapper_fold_remap_stages(struct wrapper_device *device, VkPipelineLayout layout,
                          VkPipelineShaderStageCreateInfo *stages, uint32_t count,
                          VkShaderModule *modules, uint32_t *module_count)
{
   for (uint32_t i = 0; i < count; i++) {
      size_t size = 0;
      uint32_t *code = wrapper_stride_module_code(device, stages[i].module, &size);
      if (!code) {
         WRAPPER_LOG(error, "Folded pipeline layout: no SPIR-V recorded for a shader module");
         return VK_ERROR_INITIALIZATION_FAILED;
      }
      uint32_t *remapped = wrapper_fold_remap_spirv(device, layout, code, size / 4);
      free(code);
      if (!remapped)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      VkShaderModuleCreateInfo mi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = size,
         .pCode = remapped,
      };
      VkShaderModule module;
      VkResult r = device->dispatch_table.CreateShaderModule(device->dispatch_handle, &mi,
                                                             NULL, &module);
      if (r == VK_SUCCESS)
         wrapper_stride_note_module(device, module, remapped, size);
      free(remapped);
      if (r != VK_SUCCESS)
         return r;
      modules[(*module_count)++] = module;
      stages[i].module = module;
   }
   return VK_SUCCESS;
}

static void
wrapper_fold_destroy_modules(struct wrapper_device *device, VkShaderModule *modules,
                             uint32_t count)
{
   for (uint32_t i = 0; i < count; i++) {
      wrapper_stride_forget_module(device, modules[i]);
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, modules[i], NULL);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateComputePipelines(VkDevice _device, VkPipelineCache pipelineCache,
                               uint32_t createInfoCount,
                               const VkComputePipelineCreateInfo *pCreateInfos,
                               const VkAllocationCallbacks *pAllocator,
                               VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   for (uint32_t i = 0; i < createInfoCount; i++)
      wrapper_scan_inline_shader(&pCreateInfos[i].stage);

   bool fold = false;
   for (uint32_t i = 0; device->fold_sets && i < createInfoCount; i++)
      fold |= wrapper_fold_layout_folded(device, pCreateInfos[i].layout);

   if ((!wrapper_lowers_inline_shaders(device) && !fold) || !createInfoCount)
      return device->dispatch_table.CreateComputePipelines(device->dispatch_handle,
         pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);

   VkComputePipelineCreateInfo *infos = malloc(createInfoCount * sizeof(*infos));
   VkShaderModule *modules = malloc(createInfoCount * sizeof(*modules));
   uint32_t module_count = 0;
   VkResult result = VK_SUCCESS;

   if (!infos || !modules) {
      free(infos);
      free(modules);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   VkShaderModule *fold_modules = malloc(createInfoCount * sizeof(*fold_modules));
   uint32_t fold_module_count = 0;
   memcpy(infos, pCreateInfos, createInfoCount * sizeof(*infos));
   for (uint32_t i = 0; i < createInfoCount && result == VK_SUCCESS; i++) {
      if (wrapper_lowers_maintenance5(device))
         infos[i].flags = wrapper_fold_pipeline_flags2(&infos[i].pNext, infos[i].flags);
      result = wrapper_lower_inline_shader_stage(device, &infos[i].stage,
                                                 modules, &module_count);
      if (result == VK_SUCCESS && fold && fold_modules &&
          wrapper_fold_layout_folded(device, infos[i].layout))
         result = wrapper_fold_remap_stages(device, infos[i].layout, &infos[i].stage, 1,
                                            fold_modules, &fold_module_count);
   }

   if (result == VK_SUCCESS)
      result = device->dispatch_table.CreateComputePipelines(device->dispatch_handle,
         pipelineCache, createInfoCount, infos, pAllocator, pPipelines);
   else
      for (uint32_t i = 0; i < createInfoCount; i++)
         pPipelines[i] = VK_NULL_HANDLE;

   wrapper_destroy_lowered_modules(device, modules, module_count);
   if (fold_modules)
      wrapper_fold_destroy_modules(device, fold_modules, fold_module_count);
   free(fold_modules);
   free(infos);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateGraphicsPipelines(VkDevice _device, VkPipelineCache pipelineCache,
                                uint32_t createInfoCount,
                                const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                const VkAllocationCallbacks *pAllocator,
                                VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkGraphicsPipelineCreateInfo *legacy = NULL;
   VkRenderPass *render_passes = NULL;
   void *chain_mem = NULL;
   const VkGraphicsPipelineCreateInfo *driver_infos = pCreateInfos;
   VkGraphicsPipelineCreateInfo *m5_infos = NULL;
   VkPipelineShaderStageCreateInfo *m5_stages = NULL;
   VkShaderModule *m5_modules = NULL;
   uint32_t m5_module_count = 0;

   for (uint32_t i = 0; i < createInfoCount; i++)
      for (uint32_t j = 0; j < pCreateInfos[i].stageCount; j++)
         wrapper_scan_inline_shader(&pCreateInfos[i].pStages[j]);

   if (wrapper_lowers_inline_shaders(device) && createInfoCount) {
      uint32_t stage_count = 0;
      for (uint32_t i = 0; i < createInfoCount; i++)
         stage_count += pCreateInfos[i].stageCount;

      m5_infos = malloc(createInfoCount * sizeof(*m5_infos));
      m5_stages = malloc(MAX2(stage_count, 1) * sizeof(*m5_stages));
      m5_modules = malloc(MAX2(stage_count, 1) * sizeof(*m5_modules));
      if (!m5_infos || !m5_stages || !m5_modules) {
         free(m5_infos);
         free(m5_stages);
         free(m5_modules);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }

      memcpy(m5_infos, pCreateInfos, createInfoCount * sizeof(*m5_infos));
      VkResult m5_result = VK_SUCCESS;
      uint32_t s = 0;
      for (uint32_t i = 0; i < createInfoCount; i++) {
         if (wrapper_lowers_maintenance5(device))
            m5_infos[i].flags = wrapper_fold_pipeline_flags2(&m5_infos[i].pNext,
                                                             m5_infos[i].flags);
         if (!pCreateInfos[i].stageCount)
            continue;
         memcpy(&m5_stages[s], pCreateInfos[i].pStages,
                pCreateInfos[i].stageCount * sizeof(*m5_stages));
         m5_infos[i].pStages = &m5_stages[s];
         for (uint32_t j = 0; j < pCreateInfos[i].stageCount && m5_result == VK_SUCCESS; j++)
            m5_result = wrapper_lower_inline_shader_stage(device, &m5_stages[s + j],
                                                          m5_modules, &m5_module_count);
         s += pCreateInfos[i].stageCount;
      }

      if (m5_result != VK_SUCCESS) {
         wrapper_destroy_lowered_modules(device, m5_modules, m5_module_count);
         free(m5_infos);
         free(m5_stages);
         for (uint32_t i = 0; i < createInfoCount; i++)
            pPipelines[i] = VK_NULL_HANDLE;
         return m5_result;
      }

      pCreateInfos = m5_infos;
      driver_infos = m5_infos;
   }

   VkGraphicsPipelineCreateInfo *fold_infos = NULL;
   VkPipelineShaderStageCreateInfo *fold_stages = NULL;
   VkShaderModule *fold_modules = NULL;
   uint32_t fold_module_count = 0;
   bool fold = false;
   for (uint32_t i = 0; device->fold_sets && i < createInfoCount; i++)
      fold |= wrapper_fold_layout_folded(device, pCreateInfos[i].layout);
   if (fold) {
      uint32_t stage_total = 0;
      for (uint32_t i = 0; i < createInfoCount; i++)
         stage_total += pCreateInfos[i].stageCount;
      fold_infos = malloc(createInfoCount * sizeof(*fold_infos));
      fold_stages = malloc(MAX2(stage_total, 1) * sizeof(*fold_stages));
      fold_modules = malloc(MAX2(stage_total, 1) * sizeof(*fold_modules));
      VkResult fr = (fold_infos && fold_stages && fold_modules) ? VK_SUCCESS
                                                                : VK_ERROR_OUT_OF_HOST_MEMORY;
      if (fr == VK_SUCCESS) {
         memcpy(fold_infos, pCreateInfos, createInfoCount * sizeof(*fold_infos));
         uint32_t at = 0;
         for (uint32_t i = 0; i < createInfoCount && fr == VK_SUCCESS; i++) {
            if (!wrapper_fold_layout_folded(device, fold_infos[i].layout))
               continue;
            memcpy(&fold_stages[at], fold_infos[i].pStages,
                   fold_infos[i].stageCount * sizeof(*fold_stages));
            fr = wrapper_fold_remap_stages(device, fold_infos[i].layout, &fold_stages[at],
                                           fold_infos[i].stageCount, fold_modules,
                                           &fold_module_count);
            fold_infos[i].pStages = &fold_stages[at];
            at += fold_infos[i].stageCount;
         }
      }
      if (fr != VK_SUCCESS) {
         if (fold_modules)
            wrapper_fold_destroy_modules(device, fold_modules, fold_module_count);
         free(fold_infos);
         free(fold_stages);
         free(fold_modules);
         for (uint32_t i = 0; i < createInfoCount; i++)
            pPipelines[i] = VK_NULL_HANDLE;
         if (m5_infos) {
            wrapper_destroy_lowered_modules(device, m5_modules, m5_module_count);
            free(m5_infos);
            free(m5_stages);
         }
         return fr;
      }
      pCreateInfos = fold_infos;
      driver_infos = fold_infos;
   }

   if (wrapper_lowers_core13(device) &&
       !device->dispatch_table.CmdBeginRendering && createInfoCount) {
      legacy = calloc(createInfoCount, sizeof(*legacy));
      render_passes = calloc(createInfoCount, sizeof(*render_passes));
      if (!legacy || !render_passes) {
         free(legacy);
         free(render_passes);
         if (m5_infos) {
            wrapper_destroy_lowered_modules(device, m5_modules, m5_module_count);
            free(m5_infos);
            free(m5_stages);
         }
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      memcpy(legacy, pCreateInfos, createInfoCount * sizeof(*legacy));

      for (uint32_t i = 0; i < createInfoCount; i++) {
         const VkPipelineRenderingCreateInfo *rendering =
            wrapper_find_pipeline_rendering_info(pCreateInfos[i].pNext);
         if (!rendering || pCreateInfos[i].renderPass != VK_NULL_HANDLE)
            continue;
         VkSampleCountFlagBits samples = pCreateInfos[i].pMultisampleState
            ? pCreateInfos[i].pMultisampleState->rasterizationSamples
            : VK_SAMPLE_COUNT_1_BIT;
         VkResult rp_result = wrapper_create_legacy_render_pass_for_pipeline(
            device, rendering, samples, &render_passes[i]);
         if (rp_result != VK_SUCCESS) {
            for (uint32_t j = 0; j < i; j++)
               if (render_passes[j])
                  device->dispatch_table.DestroyRenderPass(
                     device->dispatch_handle, render_passes[j], NULL);
            free(legacy);
            free(render_passes);
            if (m5_infos) {
               wrapper_destroy_lowered_modules(device, m5_modules, m5_module_count);
               free(m5_infos);
               free(m5_stages);
            }
            return rp_result;
         }
         legacy[i].renderPass = render_passes[i];
         legacy[i].subpass = 0;
         legacy[i].pNext = wrapper_chain_without(&chain_mem, pCreateInfos[i].pNext,
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
         WRAPPER_LOG(debug, "Lowering dynamic graphics pipeline %u to legacy render pass %p",
                     i, (void *)(uintptr_t)render_passes[i]);
      }
      driver_infos = legacy;
   }

   VkGraphicsPipelineCreateInfo *div_infos = NULL;
   void **div_lowerings = NULL;
   if ((device->emulate_vertex_divisor || device->emulate_zero_divisor) && createInfoCount) {
      div_infos = malloc(createInfoCount * sizeof(*div_infos));
      div_lowerings = calloc(createInfoCount, sizeof(*div_lowerings));
      if (div_infos && div_lowerings) {
         for (uint32_t i = 0; i < createInfoCount; i++)
            div_lowerings[i] = wrapper_divisor_lower(device, &driver_infos[i], &div_infos[i]);
         driver_infos = div_infos;
      }
   }

   VkResult result = VK_SUCCESS;
   bool stride_emulated = false;
   for (uint32_t i = 0; i < createInfoCount && !stride_emulated; i++)
      stride_emulated = wrapper_stride_wants(device, &driver_infos[i]);

   if (stride_emulated) {
      for (uint32_t i = 0; i < createInfoCount; i++) {
         VkResult r = wrapper_stride_wants(device, &driver_infos[i])
            ? wrapper_stride_create(device, pipelineCache, &driver_infos[i], pAllocator,
                                    &pPipelines[i])
            : device->dispatch_table.CreateGraphicsPipelines(device->dispatch_handle,
                 pipelineCache, 1, &driver_infos[i], pAllocator, &pPipelines[i]);
         if (r != VK_SUCCESS) {
            pPipelines[i] = VK_NULL_HANDLE;
            if (result == VK_SUCCESS)
               result = r;
         }
      }
   } else {
      result = device->dispatch_table.CreateGraphicsPipelines(
         device->dispatch_handle, pipelineCache, createInfoCount, driver_infos,
         pAllocator, pPipelines);
   }

   if (div_lowerings) {
      for (uint32_t i = 0; i < createInfoCount; i++) {
         if (div_infos == driver_infos)
            wrapper_divisor_record(device, div_lowerings[i], pPipelines[i]);
         free(div_lowerings[i]);
      }
   }
   free(div_lowerings);
   free(div_infos);

   if (render_passes) {
      for (uint32_t i = 0; i < createInfoCount; i++) {
         if (!render_passes[i])
            continue;
         if (pPipelines[i] == VK_NULL_HANDLE) {
            device->dispatch_table.DestroyRenderPass(device->dispatch_handle,
                                                      render_passes[i], NULL);
            continue;
         }
         struct wrapper_dynamic_pipeline *pipeline = calloc(1, sizeof(*pipeline));
         if (!pipeline) {
            /* Keep the render pass alive until device teardown rather than
             * invalidate a successfully created pipeline. */
            continue;
         }
         pipeline->pipeline = pPipelines[i];
         pipeline->render_pass = render_passes[i];
         simple_mtx_lock(&device->resource_mutex);
         _mesa_hash_table_u64_insert(device->dynamic_pipeline_table,
                                     (uint64_t)pPipelines[i], pipeline);
         simple_mtx_unlock(&device->resource_mutex);
      }
   }
   free(legacy);
   free(render_passes);
   ralloc_free(chain_mem);
   if (fold_modules)
      wrapper_fold_destroy_modules(device, fold_modules, fold_module_count);
   free(fold_infos);
   free(fold_stages);
   free(fold_modules);
   if (m5_infos) {
      wrapper_destroy_lowered_modules(device, m5_modules, m5_module_count);
      free(m5_infos);
      free(m5_stages);
   }
   WRAPPER_LOG(info, "CreateGraphicsPipelines result %d", result);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyPipeline(VkDevice _device, VkPipeline pipeline,
                        const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_dynamic_pipeline *dynamic = NULL;

   simple_mtx_lock(&device->resource_mutex);
   dynamic = _mesa_hash_table_u64_search(device->dynamic_pipeline_table,
                                         (uint64_t)pipeline);
   if (dynamic)
      _mesa_hash_table_u64_remove(device->dynamic_pipeline_table,
                                  (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);

   if (wrapper_uses_variants(device))
      wrapper_stride_destroy(device, pipeline);
   if ((device->emulate_vertex_divisor || device->emulate_zero_divisor))
      wrapper_divisor_destroy(device, pipeline);
   device->dispatch_table.DestroyPipeline(device->dispatch_handle, pipeline,
                                           pAllocator);
   if (dynamic) {
      device->dispatch_table.DestroyRenderPass(device->dispatch_handle,
                                                dynamic->render_pass, NULL);
      free(dynamic);
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
                uint32_t instanceCount, uint32_t firstVertex,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->divisor_pipeline &&
       wrapper_divisor_draw(wcb, false, vertexCount, instanceCount, firstVertex, 0,
                            firstInstance))
      return;
   if (wcb->stride_pipeline)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDraw(wcb->dispatch_handle, vertexCount,
                                       instanceCount, firstVertex, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                       uint32_t instanceCount, uint32_t firstIndex,
                       int32_t vertexOffset, uint32_t firstInstance)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->divisor_pipeline &&
       wrapper_divisor_draw(wcb, true, indexCount, instanceCount, firstIndex, vertexOffset,
                            firstInstance))
      return;
   if (wcb->stride_pipeline)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawIndexed(
      wcb->dispatch_handle, indexCount, instanceCount, firstIndex,
      vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                        VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   bool divisor = wcb->divisor_pipeline && wrapper_divisor_begin_indirect(wcb);
   if (wcb->stride_pipeline && !divisor)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawIndirect(wcb->dispatch_handle, buffer, offset,
                                               drawCount, stride);
   if (divisor)
      wrapper_divisor_end_indirect(wcb);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                               VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   bool divisor = wcb->divisor_pipeline && wrapper_divisor_begin_indirect(wcb);
   if (wcb->stride_pipeline && !divisor)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawIndexedIndirect(wcb->dispatch_handle, buffer, offset,
                                                      drawCount, stride);
   if (divisor)
      wrapper_divisor_end_indirect(wcb);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer,
                             VkDeviceSize offset, VkBuffer countBuffer,
                             VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                             uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   bool divisor = wcb->divisor_pipeline && wrapper_divisor_begin_indirect(wcb);
   if (wcb->stride_pipeline && !divisor)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawIndirectCount(wcb->dispatch_handle, buffer, offset,
      countBuffer, countBufferOffset, maxDrawCount, stride);
   if (divisor)
      wrapper_divisor_end_indirect(wcb);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                    VkDeviceSize offset, VkBuffer countBuffer,
                                    VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                    uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   bool divisor = wcb->divisor_pipeline && wrapper_divisor_begin_indirect(wcb);
   if (wcb->stride_pipeline && !divisor)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawIndexedIndirectCount(wcb->dispatch_handle, buffer,
      offset, countBuffer, countBufferOffset, maxDrawCount, stride);
   if (divisor)
      wrapper_divisor_end_indirect(wcb);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                        const VkMultiDrawInfoEXT *pVertexInfo, uint32_t instanceCount,
                        uint32_t firstInstance, uint32_t stride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->divisor_pipeline && drawCount) {
      const uint8_t *info = (const uint8_t *)pVertexInfo;
      bool handled = true;
      for (uint32_t i = 0; i < drawCount && handled; i++) {
         const VkMultiDrawInfoEXT *d = (const void *)(info + (size_t)i * stride);
         handled = wrapper_divisor_draw(wcb, false, d->vertexCount, instanceCount,
                                        d->firstVertex, 0, firstInstance);
      }
      if (handled)
         return;
   }
   if (wcb->stride_pipeline)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawMultiEXT(wcb->dispatch_handle, drawCount, pVertexInfo,
                                               instanceCount, firstInstance, stride);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                               const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                               uint32_t instanceCount, uint32_t firstInstance,
                               uint32_t stride, const int32_t *pVertexOffset)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->divisor_pipeline && drawCount) {
      const uint8_t *info = (const uint8_t *)pIndexInfo;
      bool handled = true;
      for (uint32_t i = 0; i < drawCount && handled; i++) {
         const VkMultiDrawIndexedInfoEXT *d = (const void *)(info + (size_t)i * stride);
         handled = wrapper_divisor_draw(wcb, true, d->indexCount, instanceCount, d->firstIndex,
                                        pVertexOffset ? *pVertexOffset : d->vertexOffset,
                                        firstInstance);
      }
      if (handled)
         return;
   }
   if (wcb->stride_pipeline)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawMultiIndexedEXT(wcb->dispatch_handle, drawCount,
      pIndexInfo, instanceCount, firstInstance, stride, pVertexOffset);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer, uint32_t instanceCount,
                                    uint32_t firstInstance, VkBuffer counterBuffer,
                                    VkDeviceSize counterBufferOffset,
                                    uint32_t counterOffset, uint32_t vertexStride)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->stride_pipeline)
      wrapper_stride_flush(wcb);
   wcb->device->dispatch_table.CmdDrawIndirectByteCountEXT(wcb->dispatch_handle,
      instanceCount, firstInstance, counterBuffer, counterBufferOffset, counterOffset,
      vertexStride);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_ResetCommandBuffer(VkCommandBuffer commandBuffer,
                          VkCommandBufferResetFlags flags)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wrapper_dynamic_render_objects_reset(wcb);
   wrapper_stride_reset(wcb);
   wrapper_bcn_transients_free(wcb);
   wrapper_cmd_track_reset(wcb);
   wrapper_divisor_reset(wcb);
   if (wcb->device->emulate_push_descriptor)
      wrapper_push_pool_reset_all(wcb);
   return wcb->device->dispatch_table.ResetCommandBuffer(wcb->dispatch_handle,
                                                         flags);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_ResetCommandPool(VkDevice _device, VkCommandPool commandPool,
                        VkCommandPoolResetFlags flags)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   /* Both of these walk the device's whole command-buffer list under its lock,
    * so do it only when something can have put objects there. */
   if (device->emulate_push_descriptor ||
       device->physical->emulate_imageless_framebuffer ||
       device->physical->emulate_vulkan13 ||
       device->physical->emulate_bcn > 1) {
      simple_mtx_lock(&device->resource_mutex);
      list_for_each_entry(struct wrapper_command_buffer, wcb,
                          &device->command_buffer_list, link) {
         if (wcb->pool == commandPool) {
            wrapper_dynamic_render_objects_reset(wcb);
            wrapper_bcn_transients_free(wcb);
            if (device->emulate_push_descriptor)
               wrapper_push_pool_reset_all(wcb);
         }
      }
      simple_mtx_unlock(&device->resource_mutex);
   }
   return device->dispatch_table.ResetCommandPool(device->dispatch_handle,
                                                  commandPool, flags);
}

static const char *
wrapper_fault_addr_type_str(VkDeviceFaultAddressTypeEXT t)
{
   switch (t) {
   case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT:                        return "none";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:               return "read-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:              return "write-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:            return "execute-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "ip-unknown";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "ip-invalid";
   case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:  return "ip-fault";
   default:                                                          return "unknown";
   }
}

/* On VK_ERROR_DEVICE_LOST, pull the VK_EXT_device_fault report so a black
 * screen / GPU hang shows WHERE it faulted (address + vendor codes) instead of
 * just a generic device-lost on the next submit. Emitted straight to stderr
 * under WRAPPER_DIAG -- exactly like wrapper_emit_diag's capability block -- so
 * it lands in the shared per-game diag file regardless of WRAPPER_LOG_LEVEL. A
 * device-lost is precisely when you most need this and least want it hidden
 * behind a log flag a tester forgot to set. */
static void
wrapper_log_device_fault(struct wrapper_device *device)
{
   PFN_vkGetDeviceFaultInfoEXT get_fault_info =
      device->dispatch_table.GetDeviceFaultInfoEXT;

   if (!device->device_fault_enabled || !get_fault_info)
      return;

   static int diag = -1;
   if (diag == -1)
      diag = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
   if (!diag)
      return;

   VkDeviceFaultCountsEXT counts = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
   };
   if (get_fault_info(device->dispatch_handle, &counts, NULL) != VK_SUCCESS)
      return;

   VkDeviceFaultInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
   };
   if (counts.addressInfoCount)
      info.pAddressInfos =
         calloc(counts.addressInfoCount, sizeof(VkDeviceFaultAddressInfoEXT));
   if (counts.vendorInfoCount)
      info.pVendorInfos =
         calloc(counts.vendorInfoCount, sizeof(VkDeviceFaultVendorInfoEXT));
   /* We don't consume the opaque vendor crash dump. */
   counts.vendorBinarySize = 0;

#define DF(...) fprintf(stderr, "[WRAPPER_DIAG] " __VA_ARGS__)
   if (get_fault_info(device->dispatch_handle, &counts, &info) == VK_SUCCESS) {
      DF("==== VK_ERROR_DEVICE_LOST: GPU fault report ====\n");
      DF("  description: %s\n", info.description);

      for (uint32_t i = 0; info.pAddressInfos && i < counts.addressInfoCount; i++) {
         const VkDeviceFaultAddressInfoEXT *a = &info.pAddressInfos[i];
         /* reportedAddress is only exact to addressPrecision (a power of two):
          * the faulting address lies within that aligned range. */
         DF("  address fault[%u]: type=%s reportedAddress=0x%llx precision=0x%llx\n",
            i, wrapper_fault_addr_type_str(a->addressType),
            (unsigned long long)a->reportedAddress,
            (unsigned long long)a->addressPrecision);
      }
      for (uint32_t i = 0; info.pVendorInfos && i < counts.vendorInfoCount; i++) {
         const VkDeviceFaultVendorInfoEXT *v = &info.pVendorInfos[i];
         DF("  vendor fault[%u]: %s code=0x%llx data=0x%llx\n",
            i, v->description,
            (unsigned long long)v->vendorFaultCode,
            (unsigned long long)v->vendorFaultData);
      }
      DF("================================================\n");
   }
#undef DF

   free(info.pAddressInfos);
   free(info.pVendorInfos);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_QueueSubmit(VkQueue _queue, uint32_t submitCount,
                    const VkSubmitInfo* pSubmits, VkFence fence)
{
   VK_FROM_HANDLE(wrapper_queue, queue, _queue);
   if (queue->device->query_reset_queue && vk_device_is_lost(&queue->device->vk))
      return VK_ERROR_DEVICE_LOST;
   const bool serialise = wrapper_query_reset_owns_queue(queue);
   VkSubmitInfo wrapper_submits[submitCount];
   VkCommandBuffer *command_buffers;
   VkResult result;

   struct wrapper_fence *wf = get_wrapper_fence_from_handle(queue->device, fence);

   if (serialise)
      simple_mtx_lock(&queue->device->query_reset_mutex);
   wrapper_bcn_gpu_submit(queue);
   wrapper_bcn_hq_flush(queue);
   for (int i = 0; i < submitCount; i++) {
      const VkSubmitInfo *submit_info = &pSubmits[i];
      command_buffers = malloc(sizeof(VkCommandBuffer) *
         submit_info->commandBufferCount);
      for (int j = 0; j < submit_info->commandBufferCount; j++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb,
                        submit_info->pCommandBuffers[j]);
         wcb->fence = wf;
         command_buffers[j] = wcb->dispatch_handle;
         wrapper_bcn_hq_submitted(queue, wcb);
      }
      wrapper_submits[i] = pSubmits[i];
      wrapper_submits[i].pCommandBuffers = command_buffers;
   }

   result = queue->device->dispatch_table.QueueSubmit(
      queue->dispatch_handle, submitCount, wrapper_submits, fence);
   if (serialise)
      simple_mtx_unlock(&queue->device->query_reset_mutex);

   if (result == VK_ERROR_DEVICE_LOST)
      wrapper_log_device_fault(queue->device);

   for (int i = 0; i < submitCount; i++)
      free((void *)wrapper_submits[i].pCommandBuffers);

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_QueueSubmit2(VkQueue _queue, uint32_t submitCount,
                     const VkSubmitInfo2* pSubmits, VkFence fence)
{
   VK_FROM_HANDLE(wrapper_queue, queue, _queue);
   if (queue->device->query_reset_queue && vk_device_is_lost(&queue->device->vk))
      return VK_ERROR_DEVICE_LOST;
   const bool serialise = wrapper_query_reset_owns_queue(queue);
   VkSubmitInfo2 wrapper_submits[submitCount];
   VkCommandBufferSubmitInfo *command_buffers;
   VkResult result;

   struct wrapper_fence *wf = get_wrapper_fence_from_handle(queue->device, fence);

   if (serialise)
      simple_mtx_lock(&queue->device->query_reset_mutex);
   wrapper_bcn_gpu_submit(queue);
   wrapper_bcn_hq_flush(queue);
   for (int i = 0; i < submitCount; i++) {
      const VkSubmitInfo2 *submit_info = &pSubmits[i];
      command_buffers = malloc(sizeof(VkCommandBufferSubmitInfo) *
         submit_info->commandBufferInfoCount);
      for (int j = 0; j < submit_info->commandBufferInfoCount; j++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb,
                        submit_info->pCommandBufferInfos[j].commandBuffer);
         wcb->fence = wf;
         command_buffers[j] = pSubmits[i].pCommandBufferInfos[j];
         command_buffers[j].commandBuffer = wcb->dispatch_handle;
         wrapper_bcn_hq_submitted(queue, wcb);
      }
      wrapper_submits[i] = pSubmits[i];
      wrapper_submits[i].pCommandBufferInfos = command_buffers;
   }

   if (queue->device->dispatch_table.QueueSubmit2) {
      result = queue->device->dispatch_table.QueueSubmit2(
         queue->dispatch_handle, submitCount, wrapper_submits, fence);
   } else if (!wrapper_lowers_core13(queue->device)) {
      result = VK_ERROR_FEATURE_NOT_PRESENT;
   } else {
      /* GameNative may expose a newer Vulkan core version than the Android
       * driver implements.  Newer D8VK consequently uses the Vulkan 1.3
       * vkQueueSubmit2 entry point, but Tegra's 1.1-era ICD has no function
       * pointer for it.  Calling that NULL pointer used to surface in 32-bit
       * Wine as an assertion in loader_thunks.c rather than a useful Vulkan
       * error.  Translate synchronization2 submits to the legacy operation.
       *
       * VkSemaphoreSubmitInfo carries timeline values inline whereas legacy
       * submit uses VkTimelineSemaphoreSubmitInfo.  Stage masks added by sync2
       * that cannot be represented in 32 bits are conservatively widened to
       * ALL_COMMANDS; this is slower only for that submission and preserves
       * ordering. */

      VkSubmitInfo *legacy = calloc(submitCount, sizeof(*legacy));
      VkSemaphore **waits = calloc(submitCount, sizeof(*waits));
      VkPipelineStageFlags **wait_stages =
         calloc(submitCount, sizeof(*wait_stages));
      VkCommandBuffer **commands = calloc(submitCount, sizeof(*commands));
      VkSemaphore **signals = calloc(submitCount, sizeof(*signals));
      uint64_t **wait_values = calloc(submitCount, sizeof(*wait_values));
      uint64_t **signal_values = calloc(submitCount, sizeof(*signal_values));
      VkTimelineSemaphoreSubmitInfo *timeline =
         calloc(submitCount, sizeof(*timeline));
      VkProtectedSubmitInfo *protected =
         calloc(submitCount, sizeof(*protected));

      if ((!legacy || !waits || !wait_stages || !commands || !signals ||
           !wait_values || !signal_values || !timeline || !protected) &&
          submitCount) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto legacy_submit_cleanup;
      }

      bool timeline_supported =
         queue->device->physical->base_supported_features.timelineSemaphore;

      for (uint32_t i = 0; i < submitCount; i++) {
         const VkSubmitInfo2 *src = &wrapper_submits[i];
         bool has_timeline_value = false;

         waits[i] = calloc(src->waitSemaphoreInfoCount, sizeof(**waits));
         wait_stages[i] =
            calloc(src->waitSemaphoreInfoCount, sizeof(**wait_stages));
         wait_values[i] =
            calloc(src->waitSemaphoreInfoCount, sizeof(**wait_values));
         commands[i] =
            calloc(src->commandBufferInfoCount, sizeof(**commands));
         signals[i] =
            calloc(src->signalSemaphoreInfoCount, sizeof(**signals));
         signal_values[i] =
            calloc(src->signalSemaphoreInfoCount, sizeof(**signal_values));

         if ((!waits[i] || !wait_stages[i] || !wait_values[i]) &&
             src->waitSemaphoreInfoCount) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto legacy_submit_cleanup;
         }
         if (!commands[i] && src->commandBufferInfoCount) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto legacy_submit_cleanup;
         }
         if ((!signals[i] || !signal_values[i]) &&
             src->signalSemaphoreInfoCount) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto legacy_submit_cleanup;
         }

         for (uint32_t j = 0; j < src->waitSemaphoreInfoCount; j++) {
            const VkSemaphoreSubmitInfo *sem = &src->pWaitSemaphoreInfos[j];
            waits[i][j] = sem->semaphore;
            wait_values[i][j] = sem->value;
            has_timeline_value |= sem->value != 0;
            wait_stages[i][j] = (VkPipelineStageFlags)sem->stageMask;
            if (!wait_stages[i][j] || (sem->stageMask >> 32))
               wait_stages[i][j] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
         }
         for (uint32_t j = 0; j < src->commandBufferInfoCount; j++)
            commands[i][j] = src->pCommandBufferInfos[j].commandBuffer;
         for (uint32_t j = 0; j < src->signalSemaphoreInfoCount; j++) {
            const VkSemaphoreSubmitInfo *sem = &src->pSignalSemaphoreInfos[j];
            signals[i][j] = sem->semaphore;
            signal_values[i][j] = sem->value;
            has_timeline_value |= sem->value != 0;
         }

         legacy[i] = (VkSubmitInfo) {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = src->pNext,
            .waitSemaphoreCount = src->waitSemaphoreInfoCount,
            .pWaitSemaphores = waits[i],
            .pWaitDstStageMask = wait_stages[i],
            .commandBufferCount = src->commandBufferInfoCount,
            .pCommandBuffers = commands[i],
            .signalSemaphoreCount = src->signalSemaphoreInfoCount,
            .pSignalSemaphores = signals[i],
         };

         if (has_timeline_value) {
            if (!timeline_supported) {
               WRAPPER_LOG(error,
                  "Cannot translate timeline semaphore submit: base driver lacks support");
               result = VK_ERROR_FEATURE_NOT_PRESENT;
               goto legacy_submit_cleanup;
            }
            timeline[i] = (VkTimelineSemaphoreSubmitInfo) {
               .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
               .pNext = legacy[i].pNext,
               .waitSemaphoreValueCount = src->waitSemaphoreInfoCount,
               .pWaitSemaphoreValues = wait_values[i],
               .signalSemaphoreValueCount = src->signalSemaphoreInfoCount,
               .pSignalSemaphoreValues = signal_values[i],
            };
            legacy[i].pNext = &timeline[i];
         }

         if (src->flags & VK_SUBMIT_PROTECTED_BIT) {
            protected[i] = (VkProtectedSubmitInfo) {
               .sType = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO,
               .pNext = legacy[i].pNext,
               .protectedSubmit = VK_TRUE,
            };
            legacy[i].pNext = &protected[i];
         }
      }

      result = queue->device->dispatch_table.QueueSubmit(
         queue->dispatch_handle, submitCount, legacy, fence);

legacy_submit_cleanup:
      for (uint32_t i = 0; i < submitCount; i++) {
         free(waits ? waits[i] : NULL);
         free(wait_stages ? wait_stages[i] : NULL);
         free(commands ? commands[i] : NULL);
         free(signals ? signals[i] : NULL);
         free(wait_values ? wait_values[i] : NULL);
         free(signal_values ? signal_values[i] : NULL);
      }
      free(legacy);
      free(waits);
      free(wait_stages);
      free(commands);
      free(signals);
      free(wait_values);
      free(signal_values);
      free(timeline);
      free(protected);
   }
   if (serialise)
      simple_mtx_unlock(&queue->device->query_reset_mutex);

   if (result == VK_ERROR_DEVICE_LOST)
      wrapper_log_device_fault(queue->device);

   for (int i = 0; i < submitCount; i++)
      free((void *)wrapper_submits[i].pCommandBufferInfos);

   return result;
}

static void 
wrapper_fence_destroy(struct wrapper_device *device,
					  struct wrapper_fence *wf,
					  const VkAllocationCallbacks *pAllocator)
{
   if (wf == NULL)
      return;

   simple_mtx_lock(&device->resource_mutex);

   device->dispatch_table.DestroyFence(device->dispatch_handle,
      wf->dispatch_handle, pAllocator); 

   _mesa_hash_table_u64_remove(device->fence_table, (uint64_t)wf->dispatch_handle);
   list_del(&wf->link);
   
   simple_mtx_unlock(&device->resource_mutex);
   
   vk_object_free(&device->vk, &device->vk.alloc, wf);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateFence(VkDevice _device,
					const VkFenceCreateInfo *pCreateInfo,
					const VkAllocationCallbacks *pAllocator,
					VkFence *pFence)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   VkResult res = device->dispatch_table.CreateFence(device->dispatch_handle,
      pCreateInfo, pAllocator, pFence);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create fence, res %d", res);
      return res;
   }

   simple_mtx_lock(&device->resource_mutex);
   
   struct wrapper_fence *wf = vk_object_zalloc(&device->vk,
      &device->vk.alloc, sizeof(struct wrapper_fence), VK_OBJECT_TYPE_FENCE);
   if (!wf) {
      WRAPPER_LOG(error, "Failed to allocate wrapper_fence");
      simple_mtx_unlock(&device->resource_mutex);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   wf->device = device;
   wf->dispatch_handle = *pFence;

   list_add(&wf->link, &device->fence_list);
   _mesa_hash_table_u64_insert(device->fence_table, (uint64_t)wf->dispatch_handle, wf);

   simple_mtx_unlock(&device->resource_mutex);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_WaitForFences(VkDevice _device,
					  uint32_t fenceCount,
					  const VkFence *pFences,
					  VkBool32 waitAll,
					  uint64_t timeout)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult res;

   res = device->dispatch_table.WaitForFences(device->dispatch_handle,
     fenceCount, pFences, waitAll, timeout);

   return res;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyFence(VkDevice _device,
					 VkFence fence,
					 const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   struct wrapper_fence *wf = get_wrapper_fence_from_handle(device, fence);
   wrapper_fence_destroy(device, wf, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdExecuteCommands(VkCommandBuffer commandBuffer,
                           uint32_t commandBufferCount,
                           const VkCommandBuffer* pCommandBuffers)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   VkCommandBuffer command_buffers[commandBufferCount];

   for (int i = 0; i < commandBufferCount; i++) {
      command_buffers[i] =
         wrapper_command_buffer_from_handle(pCommandBuffers[i])->dispatch_handle;
   }
   wcb->device->dispatch_table.CmdExecuteCommands(
      wcb->dispatch_handle, commandBufferCount, command_buffers);
}

static uint32_t *
wrapper_patch_spirv(struct wrapper_device *device, const void *code, size_t *size)
{
   static int wrapper_no_remove_clip_distance = -1;
   static int wrapper_no_patch_OpConstComp = -1;

   if (!wrapper_patches_spirv(device))
      return NULL;

   if (wrapper_no_remove_clip_distance == -1)
      wrapper_no_remove_clip_distance = getenv("WRAPPER_NO_REMOVE_CLIP_DISTANCE") && atoi(getenv("WRAPPER_NO_REMOVE_CLIP_DISTANCE"));

   if (wrapper_no_patch_OpConstComp == -1)
      wrapper_no_patch_OpConstComp = getenv("WRAPPER_NO_PATCH_OPCONSTCOMP") && atoi(getenv("WRAPPER_NO_PATCH_OPCONSTCOMP"));

   uint32_t *patched = malloc(*size);
   if (!patched)
      return NULL;
   memcpy(patched, code, *size);
   if (!wrapper_no_patch_OpConstComp)
      patch_OpConstantComposite_to_OpSpecConstantComposite(patched, *size / sizeof(uint32_t));
   if (!wrapper_no_remove_clip_distance)
      remove_ClipDistance(patched, size);
   return patched;
}

static uint32_t
wrapper_max_spirv_version(const struct wrapper_physical_device *pdevice)
{
   uint32_t api = pdevice->properties2.properties.apiVersion;
   if (api >= VK_API_VERSION_1_3)
      return 0x00010600;
   if (api >= VK_API_VERSION_1_2)
      return 0x00010500;
   if (pdevice->base_supported_extensions.KHR_spirv_1_4)
      return 0x00010400;
   return 0x00010300;
}

static uint32_t *
wrapper_lower_spirv(struct wrapper_device *device, const uint32_t *code, size_t *size)
{
   if (!wrapper_lowers_core13(device) || *size < 20)
      return NULL;
   bool lower_demote = !device->physical->base_supported_features.shaderDemoteToHelperInvocation;
   if (code[1] <= wrapper_max_spirv_version(device->physical) && !lower_demote)
      return NULL;
   uint32_t flags = device->physical->base_supported_features.shaderZeroInitializeWorkgroupMemory
      ? 0 : WRAPPER_SPIRV_STRIP_WORKGROUP_INIT;
   if (lower_demote)
      flags |= WRAPPER_SPIRV_LOWER_DEMOTE;
   uint32_t *lowered = NULL;
   size_t lowered_words = 0;
   enum wrapper_spirv_lower_result r =
      wrapper_spirv_lower_to_1_3(code, *size / 4, flags, &lowered, &lowered_words);
   if (r == WRAPPER_SPIRV_LOWERED) {
      *size = lowered_words * 4;
      return lowered;
   }
   if (r == WRAPPER_SPIRV_FAILED)
      WRAPPER_LOG(error, "Cannot lower SPIR-V %u.%u to 1.3 for this driver",
                  (code[1] >> 16) & 0xff, (code[1] >> 8) & 0xff);
   free(lowered);
   return NULL;
}

static VkResult
wrapper_lower_int64_module(const VkShaderModuleCreateInfo *info,
                           VkShaderModuleCreateInfo *out_info, uint32_t **out_code)
{
   uint32_t *code = NULL;
   size_t words = 0;
   enum wrapper_spirv_lower_result r =
      wrapper_spirv_lower_int64(info->pCode, info->codeSize / 4, &code, &words);
   if (r == WRAPPER_SPIRV_LOWERED) {
      *out_info = *info;
      out_info->pCode = code;
      out_info->codeSize = words * 4;
      *out_code = code;
      return VK_SUCCESS;
   }
   if (r == WRAPPER_SPIRV_UNCHANGED)
      return VK_SUCCESS;

   uint64_t hash = 0xcbf29ce484222325ull;
   const uint8_t *bytes = (const uint8_t *)info->pCode;
   for (size_t i = 0; i < info->codeSize; i++)
      hash = (hash ^ bytes[i]) * 0x100000001b3ull;
   const char *name = "";
   for (size_t i = 5; i < info->codeSize / 4 && (info->pCode[i] >> 16);
        i += info->pCode[i] >> 16) {
      if ((info->pCode[i] & 0xffff) == 7 && (info->pCode[i] >> 16) > 2) {
         name = (const char *)&info->pCode[i + 2];
         break;
      }
   }
   WRAPPER_LOG(info, "Refusing shader %.64s (spirv hash %016llx): it uses 64-bit integers "
               "beyond widening before a float conversion and the driver lacks shaderInt64",
               name, (unsigned long long)hash);
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateShaderModule(VkDevice _device,
						   const VkShaderModuleCreateInfo *pCreateInfo,
						   const VkAllocationCallbacks *pAllocator,
						   VkShaderModule *pShaderModule)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   bcn_scan_shader(pCreateInfo->pCode, pCreateInfo->codeSize);

   uint32_t *int64_code = NULL;
   VkShaderModuleCreateInfo int64_info;
   if (device->physical->lower_int64) {
      VkResult ir = wrapper_lower_int64_module(pCreateInfo, &int64_info, &int64_code);
      if (ir != VK_SUCCESS)
         return ir;
      if (int64_code)
         pCreateInfo = &int64_info;
   }


   VkShaderModuleCreateInfo create_info = *pCreateInfo;
   uint32_t *lowered = wrapper_lower_spirv(device, pCreateInfo->pCode, &create_info.codeSize);
   if (lowered)
      create_info.pCode = lowered;
   uint32_t *patched = wrapper_patch_spirv(device, create_info.pCode, &create_info.codeSize);
   if (patched)
      create_info.pCode = patched;

   if (WRAPPER_LOG_LEVEL(shader))
      dump_shader_code(create_info.pCode, create_info.codeSize);
   
   VkResult result = device->dispatch_table.CreateShaderModule(
      device->dispatch_handle, &create_info, pAllocator, pShaderModule);
   if (result == VK_SUCCESS && wrapper_tracks_shader_code(device))
      wrapper_stride_note_module(device, *pShaderModule, create_info.pCode,
                                 create_info.codeSize);
   free(patched);
   free(lowered);
   free(int64_code);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateShadersEXT(VkDevice _device, uint32_t createInfoCount,
                         const VkShaderCreateInfoEXT *pCreateInfos,
                         const VkAllocationCallbacks *pAllocator,
                         VkShaderEXT *pShaders)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkShaderCreateInfoEXT *infos = NULL;
   uint32_t **patched = NULL;
   uint32_t **lowered = NULL;

   for (uint32_t i = 0; i < createInfoCount; i++)
      if (pCreateInfos[i].codeType == VK_SHADER_CODE_TYPE_SPIRV_EXT)
         bcn_scan_shader(pCreateInfos[i].pCode, pCreateInfos[i].codeSize);

   if ((wrapper_patches_spirv(device) || wrapper_lowers_core13(device)) && createInfoCount) {
      infos = malloc(createInfoCount * sizeof(*infos));
      patched = calloc(createInfoCount, sizeof(*patched));
      lowered = calloc(createInfoCount, sizeof(*lowered));
      if (!infos || !patched || !lowered) {
         free(infos);
         free(patched);
         free(lowered);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      memcpy(infos, pCreateInfos, createInfoCount * sizeof(*infos));
      for (uint32_t i = 0; i < createInfoCount; i++) {
         if (infos[i].codeType != VK_SHADER_CODE_TYPE_SPIRV_EXT)
            continue;
         lowered[i] = wrapper_lower_spirv(device, infos[i].pCode, &infos[i].codeSize);
         if (lowered[i])
            infos[i].pCode = lowered[i];
         patched[i] = wrapper_patch_spirv(device, infos[i].pCode, &infos[i].codeSize);
         if (patched[i])
            infos[i].pCode = patched[i];
      }
      pCreateInfos = infos;
   }

   VkResult result = device->dispatch_table.CreateShadersEXT(
      device->dispatch_handle, createInfoCount, pCreateInfos, pAllocator, pShaders);

   if (patched)
      for (uint32_t i = 0; i < createInfoCount; i++) {
         free(patched[i]);
         free(lowered[i]);
      }
   free(patched);
   free(lowered);
   free(infos);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyShaderModule(VkDevice _device, VkShaderModule shaderModule,
                            const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   if (wrapper_tracks_shader_code(device))
      wrapper_stride_forget_module(device, shaderModule);
   device->dispatch_table.DestroyShaderModule(device->dispatch_handle, shaderModule,
                                              pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBindPipeline(VkCommandBuffer commandBuffer,
                        VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   if (wcb->device->bcn_bda)
      wrapper_cmd_track_pipeline(wcb, pipelineBindPoint, pipeline);
   if (wrapper_uses_variants(wcb->device) &&
       pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
      wrapper_stride_bind(wcb, pipeline);
   if ((wcb->device->emulate_vertex_divisor || wcb->device->emulate_zero_divisor) &&
       pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
      wrapper_divisor_bind(wcb, pipeline);
   wcb->device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle, pipelineBindPoint,
                                               pipeline);
}						   						   

static VkResult
wrapper_command_buffer_create(struct wrapper_device *device,
                              VkCommandPool pool,
                              VkCommandBuffer dispatch_handle,
                              VkCommandBuffer *pCommandBuffers) {
   struct wrapper_command_buffer *wcb;
   wcb = vk_object_zalloc(&device->vk, &device->vk.alloc,
                          sizeof(struct wrapper_command_buffer),
                          VK_OBJECT_TYPE_COMMAND_BUFFER);
   if (!wcb)
      return vk_error(&device->vk, VK_ERROR_OUT_OF_HOST_MEMORY);

   wcb->device = device;
   wcb->pool = pool;
   wcb->dispatch_handle = dispatch_handle;
   list_inithead(&wcb->dynamic_render_objects);
   list_inithead(&wcb->bcn_transients);
   list_add(&wcb->link, &device->command_buffer_list);

   *pCommandBuffers = wrapper_command_buffer_to_handle(wcb);

   return VK_SUCCESS;
}

static void
wrapper_command_buffer_destroy(struct wrapper_device *device,
                               struct wrapper_command_buffer *wcb) {
   if (wcb == NULL)
      return;

   if (device->emulate_push_descriptor)
      wrapper_push_pool_destroy_all(wcb);
   wrapper_dynamic_render_objects_reset(wcb);
   wrapper_bcn_transients_free(wcb);

   device->dispatch_table.FreeCommandBuffers(
      device->dispatch_handle, wcb->pool, 1, &wcb->dispatch_handle);

   list_del(&wcb->link);
   vk_object_free(&device->vk, &device->vk.alloc, wcb);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateCommandBuffers(VkDevice _device,
                               const VkCommandBufferAllocateInfo* pAllocateInfo,
                               VkCommandBuffer* pCommandBuffers)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult result;
   uint32_t i;
   
   result = device->dispatch_table.AllocateCommandBuffers(
      device->dispatch_handle, pAllocateInfo, pCommandBuffers);
   WRAPPER_LOG(info, "AllocateCommandBuffers driver result %d count=%u",
               result, pAllocateInfo->commandBufferCount);
   if (result != VK_SUCCESS)
      return result;

   simple_mtx_lock(&device->resource_mutex);

   for (i = 0; i < pAllocateInfo->commandBufferCount; i++) {
      result = wrapper_command_buffer_create(
         device, pAllocateInfo->commandPool, pCommandBuffers[i],
         pCommandBuffers + i);
      if (result != VK_SUCCESS)
         break;
      wrapper_command_buffer_from_handle(pCommandBuffers[i])->secondary =
         pAllocateInfo->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY;
   }

   if (result != VK_SUCCESS) {
      for (int q = 0; q < i; q++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb, pCommandBuffers[q]);
         wrapper_command_buffer_destroy(device, wcb);
      }

      device->dispatch_table.FreeCommandBuffers(
         device->dispatch_handle, pAllocateInfo->commandPool,
         pAllocateInfo->commandBufferCount - i, pCommandBuffers + i);
      
      for (i = 0; i < pAllocateInfo->commandBufferCount; i++) {
         pCommandBuffers[i] = VK_NULL_HANDLE;
      }
   }

   simple_mtx_unlock(&device->resource_mutex);

   return result;
}

/* Dropped (capped) mips skip the transcode but still leave their .src. */
static void
wrapper_bcn_note_dropped(struct wrapper_device *device, struct wrapper_buffer *wb,
                         VkFormat format, uint32_t mip_drop, uint32_t regionCount,
                         const VkBufferImageCopy *pRegions)
{
   if (!bcn_cache_enabled() || !bcn_upload_enabled() || wb->device_only)
      return;

   simple_mtx_lock(&device->resource_mutex);
   bool was_mapped = wb->is_mapped;
   if (!was_mapped) {
      if (device->dispatch_table.MapMemory(device->dispatch_handle, wb->memory,
            wb->offset, wb->size, 0, &wb->mapped_address) != VK_SUCCESS) {
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }
      wb->is_mapped = 1;
   }

   for (uint32_t i = 0; i < regionCount; i++) {
      const VkBufferImageCopy *r = &pRegions[i];
      if (r->imageSubresource.mipLevel >= mip_drop)
         continue;
      int w = r->imageExtent.width;
      bcn_cache_note_source(wb->mapped_address, w, r->imageExtent.height,
         r->bufferRowLength ? (int)r->bufferRowLength : w, format,
         (int)r->bufferOffset);
   }

   if (!was_mapped) {
      device->dispatch_table.UnmapMemory(device->dispatch_handle, wb->memory);
      wb->is_mapped = 0;
   }
   simple_mtx_unlock(&device->resource_mutex);
}

static uint32_t
wrapper_bcn_region_slices(const VkBufferImageCopy *r)
{
   return MAX2(r->imageSubresource.layerCount, 1) * MAX2(r->imageExtent.depth, 1);
}

static int
wrapper_bcn_block_bytes(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
   case VK_FORMAT_BC4_UNORM_BLOCK:
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return 8;
   default:
      return 16;
   }
}

/* Mapped host-visible staging buffer for one upload; NULL on failure. */
static struct wrapper_buffer *
wrapper_bcn_staging_create(struct wrapper_device *device, VkDeviceSize upload_size)
{
   VkResult res;
   struct wrapper_buffer *staging_wb = vk_object_zalloc(&device->vk,
      &device->vk.alloc, sizeof(struct wrapper_buffer), VK_OBJECT_TYPE_BUFFER);

   VkBufferCreateInfo buffer_create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = upload_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .flags = 0,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   res = device->dispatch_table.CreateBuffer(device->dispatch_handle,
      &buffer_create_info, NULL, &staging_wb->dispatch_handle);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create staging buffer, res %d", res);
      return NULL;
   }

   VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = upload_size,
      .memoryTypeIndex = wrapper_select_device_memory_type(device,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
   };

   res = device->dispatch_table.AllocateMemory(device->dispatch_handle,
      &allocate_info, NULL, &staging_wb->memory);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate staging buffer memory, res %d", res);
      return NULL;
   }

   res = device->dispatch_table.BindBufferMemory(device->dispatch_handle,
      staging_wb->dispatch_handle, staging_wb->memory, 0);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to bind staging buffer memory, res %d", res);
      return NULL;
   }

   res = device->dispatch_table.MapMemory(device->dispatch_handle,
      staging_wb->memory, 0, upload_size, 0, &staging_wb->mapped_address);

   if (res != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to map staging buffer memory, res %d", res);
      return NULL;
   }

   return staging_wb;
}

/* Copy a tightly packed staging upload into the image; freed with the command
 * buffer. */
static void
wrapper_bcn_staging_submit(struct wrapper_command_buffer *wcb,
                           struct wrapper_device *device,
                           struct wrapper_buffer *staging_wb, VkImage dstImage,
                           VkImageLayout dstLayout, VkBufferImageCopy copy_region)
{
   copy_region.bufferOffset = 0;
   copy_region.bufferRowLength = 0;
   copy_region.bufferImageHeight = 0;

   device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle,
      staging_wb->dispatch_handle, dstImage, dstLayout, 1, &copy_region);

   staging_wb->wcb = wcb;
   staging_wb->device = device;
   list_addtail(&staging_wb->link, &wcb->bcn_transients);
}

static void
wrapper_bcn_cpu_copy_regions(struct wrapper_command_buffer *wcb,
                    struct wrapper_device *device,
                    struct wrapper_buffer *wb,
                    VkImage dstImage,
                    VkImageLayout dstLayout,
                    VkFormat format,
                    uint32_t regionCount,
                    const VkBufferImageCopy *pRegions)
{
   VkResult res;

   if (wb->device_only) {
      static bool logged;
      if (!logged)
         WRAPPER_LOG(error, "BCn: upload from memory the host cannot map needs the GPU path");
      logged = true;
      return;
   }

   simple_mtx_lock(&device->resource_mutex);

   if (!wb->is_mapped) {
      res = device->dispatch_table.MapMemory(device->dispatch_handle,
         wb->memory, wb->offset, wb->size, 0, &wb->mapped_address);

      if (res != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to map source buffer memory, res %d", res);
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }

      wb->is_mapped = 1;
   }

   for (int i = 0; i < regionCount; i++) {
      VkBufferImageCopy copy_region = pRegions[i];
      int w = copy_region.imageExtent.width;
      int h = copy_region.imageExtent.height;
      int offset = copy_region.bufferOffset;
      uint32_t slices = wrapper_bcn_region_slices(&copy_region);
      VkDeviceSize slice_size = bcn_upload_size(format, w, h);
      VkDeviceSize upload_size = slice_size * slices;

      struct wrapper_buffer *staging_wb = wrapper_bcn_staging_create(device, upload_size);
      if (!staging_wb) {
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }

      int src_w = copy_region.bufferRowLength ? (int)copy_region.bufferRowLength : w;
      int src_h = copy_region.bufferImageHeight ? (int)copy_region.bufferImageHeight : h;
      VkDeviceSize src_slice = (VkDeviceSize)((src_w + 3) / 4) * ((src_h + 3) / 4) *
         wrapper_bcn_block_bytes(format);
      __atomic_add_fetch(&wrapper_bcn_fg_busy, 1, __ATOMIC_RELAXED);
      for (uint32_t s = 0; s < slices; s++)
         decompress_bcn_format(wb->mapped_address,
            (char *)staging_wb->mapped_address + s * slice_size, w, h, src_w, format,
            (int)(offset + s * src_slice));
      __atomic_sub_fetch(&wrapper_bcn_fg_busy, 1, __ATOMIC_RELAXED);

      /* Texture-path diagnostic (part of WRAPPER_DIAG=1): for each BCn CPU upload,
       * record the real parameters (source stride, target format, image tiling/
       * extent/usage) and dump the leading source BC7 bytes + our transcode
       * output, so the transcode can be verified byte-for-byte offline against
       * the reference decoder and any stride/format/tiling mismatch is visible. */
      {
         static int tex_diag = -1;
         if (tex_diag == -1)
            tex_diag = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
         static int tex_n = 0;
         if (tex_diag && (!wrapper_diag_capture || tex_n < 128)) {
            struct wrapper_image *twi = get_wrapper_image_from_handle_locked(device, dstImage);
            VkFormat tgt = get_format_for_bcn(format);
            /* Write to stderr, which for a D3D process is redirected into the
             * diag file (wrapper_emit_diag), so this lands in the one file. */
            fprintf(stderr,
               "[WRAPPER_TEX %d] bc=%d %dx%d mip=%u rowLen=%u imgH=%u off=%d srcStride=%d"
               " -> target=%d uploadSize=%llu",
               tex_n, format, w, h, copy_region.imageSubresource.mipLevel,
               copy_region.bufferRowLength, copy_region.bufferImageHeight,
               offset, src_w, tgt, (unsigned long long)upload_size);
            if (twi)
               fprintf(stderr, " | img: fmt=%d extent=%ux%ux%u mips=%u tiling=%d usage=0x%x flags=0x%x",
                  twi->info.format, twi->info.extent.width, twi->info.extent.height,
                  twi->info.extent.depth, twi->info.mipLevels, twi->info.tiling,
                  twi->info.usage, twi->info.flags);
            fprintf(stderr, "\n");
            if (tex_n < 4 && !wrapper_diag_capture) {
               unsigned char *src = (unsigned char *)wb->mapped_address + offset;
               unsigned char *out = (unsigned char *)staging_wb->mapped_address;
               int ns = 96, no = (upload_size < 256) ? (int)upload_size : 256;
               fprintf(stderr, "  src:");
               for (int b = 0; b < ns; b++) fprintf(stderr, "%02x", src[b]);
               fprintf(stderr, "\n  out:");
               for (int b = 0; b < no; b++) fprintf(stderr, "%02x", out[b]);
               fprintf(stderr, "\n");
            }
            tex_n++;
            if (wrapper_diag_capture && tex_n == 128)
               fprintf(stderr, "[WRAPPER_DIAG] CPU texture record limit reached\n");
         }
      }

      wrapper_bcn_staging_submit(wcb, device, staging_wb, dstImage, dstLayout,
                                 copy_region);
   }

   if (wb->is_mapped) {
      device->dispatch_table.UnmapMemory(device->dispatch_handle,
         wb->memory);

      wb->is_mapped = 0;
   }

   simple_mtx_unlock(&device->resource_mutex);
}

static void
wrapper_bcn_do_copy_regions(struct wrapper_command_buffer *wcb,
                    struct wrapper_device *device,
                    struct wrapper_buffer *wb,
                    struct wrapper_image *wi,
                    VkImageLayout dstLayout,
                    uint32_t regionCount,
                    const VkBufferImageCopy *pRegions)
{
   bool gpu = wrapper_bcn_gpu_path();
   for (uint32_t i = 0; i < regionCount; i++) {
      /* the GPU path, or a region that has to be merged into whole ASTC blocks */
      if ((gpu || wb->device_only || wrapper_bcn_gpu_misaligned(wi, &pRegions[i])) &&
          wrapper_bcn_gpu_upload(wcb, wi, wb, dstLayout, 1, &pRegions[i]))
         continue;
      wrapper_bcn_hq_write(device, wi, &pRegions[i].imageSubresource, pRegions[i].imageOffset,
                           pRegions[i].imageExtent);
      wrapper_bcn_cpu_copy_regions(wcb, device, wb, wi->dispatch_handle, dstLayout,
                                   wi->info.format, 1, &pRegions[i]);
   }
}

/* Capped images: dropped mips leave only their .src, the rest move down,
 * transcoded in stack-sized chunks. */
static void
wrapper_bcn_do_copy(struct wrapper_command_buffer *wcb,
                    struct wrapper_device *device,
                    struct wrapper_buffer *wb,
                    struct wrapper_image *wi,
                    VkImageLayout dstLayout,
                    uint32_t regionCount,
                    const VkBufferImageCopy *pRegions)
{
   uint32_t mip_drop = wi->mip_drop;
   VkFormat format = wi->info.format;
   if (!mip_drop) {
      wrapper_bcn_do_copy_regions(wcb, device, wb, wi, dstLayout,
         regionCount, pRegions);
      return;
   }

   VkBufferImageCopy capped[WRAPPER_CAP_STACK_ENTRIES];
   bool dropped = false;
   for (uint32_t i = 0; i < regionCount; i += WRAPPER_CAP_STACK_ENTRIES) {
      uint32_t chunk = MIN2(regionCount - i, WRAPPER_CAP_STACK_ENTRIES);
      uint32_t kept = bcn_cap_copy_regions(mip_drop, pRegions + i, chunk, capped);
      dropped |= kept != chunk;
      if (kept)
         wrapper_bcn_cpu_copy_regions(wcb, device, wb, wi->dispatch_handle, dstLayout, format,
            kept, capped);
   }
   if (dropped)
      wrapper_bcn_note_dropped(device, wb, format, mip_drop, regionCount, pRegions);
}

/* Append a line to the per-game diag file and mirror to logcat, when
 * WRAPPER_DIAG is set. Used for the copy-level texture log below. */
int
wrapper_diag_on(void)
{
   if (wrapper_diag_capture)
      return 1;
   static int on = -1;
   if (on == -1)
      on = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
   return on;
}

void
wrapper_diag_append(const char *fmt, ...)
{
   if (!wrapper_diag_on())
      return;
   /* Bound high-volume resource records, retaining capability/fault/status
    * messages even after the texture budget is exhausted. */
   if (wrapper_diag_capture &&
       (!strncmp(fmt, "[COPY]", 6) || !strncmp(fmt, "[FMT]", 5))) {
      static unsigned records;
      unsigned n = __atomic_fetch_add(&records, 1, __ATOMIC_RELAXED);
      if (n >= 512) {
         if (n == 512)
            fprintf(stderr, "[WRAPPER_DIAG] format/copy record limit reached\n");
         return;
      }
   }
   /* BCn status lines also go to the stderr the process had before it was
    * redirected into the diag file, where the launcher log sees them */
   if (wrapper_diag_stderr >= 0 && !strncmp(fmt, "[BCN]", 5)) {
      va_list ap;
      va_start(ap, fmt);
      vdprintf(wrapper_diag_stderr, fmt, ap);
      va_end(ap);
   }
   const char *aid = getenv("WRAPPER_DIAG_APPID");
   char tp[256];
   snprintf(tp, sizeof(tp),
      "/data/data/app.gamenative/files/imagefs/usr/tmp/wrapper_diag_%s.txt",
      (aid && aid[0]) ? aid : "unknown");
   FILE *f = fopen(tp, "a");
   va_list ap;
   va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
   if (f) { va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap); fclose(f); }
}

/* Log every buffer->image copy that targets a tracked image (any format,
 * emulated or not) so nothing that could corrupt is invisible. */
static void
wrapper_diag_log_copy(struct wrapper_device *device, VkImage dstImage,
                      struct wrapper_image *wi, struct wrapper_buffer *wb,
                      uint32_t w, uint32_t h, uint32_t mip, uint32_t rowLen,
                      uint32_t regionCount)
{
   if (!wi)
      return;
   VkFormat req = wi->info.format;
   int emu = is_emulated_bcn(device->physical, req);
   wrapper_diag_append(
      "[COPY] img=%04x fmt=%d emulated=%d -> stored=%d %ux%u mip=%u rowLen=%u"
      " tiling=%d usage=0x%x mips=%u bufTracked=%d regions=%u\n",
      (unsigned)((uintptr_t)dstImage & 0xffff), req, emu,
      emu ? get_format_for_bcn(req) : req, w, h, mip, rowLen,
      (int)wi->info.tiling, wi->info.usage, wi->info.mipLevels,
      wb ? 1 : 0, regionCount);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyBufferToImage(VkCommandBuffer commandBuffer,
							 VkBuffer srcBuffer,
							 VkImage dstImage,
							 VkImageLayout dstLayout,
							 uint32_t regionCount,
							 const VkBufferImageCopy *pRegions)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   struct wrapper_image *wi = get_wrapper_image_from_handle(device, dstImage);
   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle(device, srcBuffer);
   VkFormat format = wi ? wi->info.format : VK_FORMAT_UNDEFINED;

   if (regionCount)
      wrapper_diag_log_copy(device, dstImage, wi, wb,
         pRegions[0].imageExtent.width, pRegions[0].imageExtent.height,
         pRegions[0].imageSubresource.mipLevel, pRegions[0].bufferRowLength,
         regionCount);

   if (!wi || !wb || !is_emulated_bcn(device->physical, format)) {
      device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle,
         srcBuffer, dstImage, dstLayout, regionCount, pRegions);
      return;
   }

   wrapper_bcn_do_copy(wcb, device, wb, wi, dstLayout, regionCount, pRegions);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer,
                             const VkCopyBufferToImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   struct wrapper_image *wi = get_wrapper_image_from_handle(device, pInfo->dstImage);
   struct wrapper_buffer *wb = get_wrapper_buffer_from_handle(device, pInfo->srcBuffer);
   VkFormat format = wi ? wi->info.format : VK_FORMAT_UNDEFINED;

   if (pInfo->regionCount)
      wrapper_diag_log_copy(device, pInfo->dstImage, wi, wb,
         pInfo->pRegions[0].imageExtent.width, pInfo->pRegions[0].imageExtent.height,
         pInfo->pRegions[0].imageSubresource.mipLevel,
         pInfo->pRegions[0].bufferRowLength, pInfo->regionCount);

   if (!wi || !wb || !is_emulated_bcn(device->physical, format)) {
      device->dispatch_table.CmdCopyBufferToImage2(wcb->dispatch_handle, pInfo);
      return;
   }

   VkBufferImageCopy regions[pInfo->regionCount];
   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pInfo->pRegions[i];
      regions[i] = (VkBufferImageCopy){
         .bufferOffset = r->bufferOffset,
         .bufferRowLength = r->bufferRowLength,
         .bufferImageHeight = r->bufferImageHeight,
         .imageSubresource = r->imageSubresource,
         .imageOffset = r->imageOffset,
         .imageExtent = r->imageExtent,
      };
   }

   wrapper_bcn_do_copy(wcb, device, wb, wi, pInfo->dstImageLayout,
      pInfo->regionCount, regions);
}

VKAPI_ATTR void VKAPI_CALL 
wrapper_CmdBlitImage(
    VkCommandBuffer commandBuffer,
    VkImage srcImage, VkImageLayout srcImageLayout,
    VkImage dstImage, VkImageLayout dstImageLayout,
    uint32_t regionCount, const VkImageBlit* pRegions,
    VkFilter filter)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   struct wrapper_image *dst_img = get_wrapper_image_from_handle(device, dstImage);
   const VkImageBlit *regions = (const VkImageBlit *)pRegions;

   if (dst_img && dst_img->is_emulated_bgra8) {
      WRAPPER_LOG(error, "vkCmdBlitImage with is_emulated_bgra8 image");
   }

   uint32_t src_drop = wrapper_image_mip_drop(device, srcImage);
   uint32_t dst_drop = dst_img ? dst_img->mip_drop : 0;
   VkImageBlit stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageBlit *capped = NULL;
   if ((src_drop || dst_drop) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), regionCount))) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < regionCount; i++) {
         capped[n] = pRegions[i];
         if (wrapper_cap_level(src_drop, &capped[n].srcSubresource.mipLevel) &&
             wrapper_cap_level(dst_drop, &capped[n].dstSubresource.mipLevel))
            n++;
      }
      if (!n) {
         wrapper_cap_free(capped, stack);
         return;
      }
      regionCount = n;
      regions = capped;
   }

   device->dispatch_table.CmdBlitImage(
      wcb->dispatch_handle,
      srcImage, srcImageLayout,
      dstImage, dstImageLayout,
      regionCount, regions, filter);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBlitImage2(
    VkCommandBuffer commandBuffer,
    const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   struct wrapper_image *dst_img = get_wrapper_image_from_handle(device, pBlitImageInfo->dstImage);

   if (dst_img && dst_img->is_emulated_bgra8) {
      WRAPPER_LOG(error, "vkCmdBlitImage2 with is_emulated_bgra8 image");
   }

   uint32_t src_drop = wrapper_image_mip_drop(device, pBlitImageInfo->srcImage);
   uint32_t dst_drop = dst_img ? dst_img->mip_drop : 0;
   VkBlitImageInfo2 capped_info;
   VkImageBlit2 stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageBlit2 *capped = NULL;
   if ((src_drop || dst_drop) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), pBlitImageInfo->regionCount))) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < pBlitImageInfo->regionCount; i++) {
         capped[n] = pBlitImageInfo->pRegions[i];
         if (wrapper_cap_level(src_drop, &capped[n].srcSubresource.mipLevel) &&
             wrapper_cap_level(dst_drop, &capped[n].dstSubresource.mipLevel))
            n++;
      }
      if (!n) {
         wrapper_cap_free(capped, stack);
         return;
      }
      capped_info = *pBlitImageInfo;
      capped_info.regionCount = n;
      capped_info.pRegions = capped;
      pBlitImageInfo = &capped_info;
   }

   if (device->dispatch_table.CmdBlitImage2) {
      device->dispatch_table.CmdBlitImage2(wcb->dispatch_handle, pBlitImageInfo);
   }
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdPipelineBarrier(VkCommandBuffer commandBuffer,
                           VkPipelineStageFlags srcStageMask,
                           VkPipelineStageFlags dstStageMask,
                           VkDependencyFlags dependencyFlags,
                           uint32_t memoryBarrierCount,
                           const VkMemoryBarrier *pMemoryBarriers,
                           uint32_t bufferMemoryBarrierCount,
                           const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                           uint32_t imageMemoryBarrierCount,
                           const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkImageMemoryBarrier stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageMemoryBarrier *capped = NULL;

   if (imageMemoryBarrierCount &&
       __atomic_load_n(&wrapper_capped_images, __ATOMIC_RELAXED) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), imageMemoryBarrierCount))) {
      imageMemoryBarrierCount = wrapper_cap_image_barriers(device,
         pImageMemoryBarriers, imageMemoryBarrierCount, capped);
      pImageMemoryBarriers = capped;
   }

   device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle, srcStageMask,
      dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers,
      bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount,
      pImageMemoryBarriers);
   wrapper_bcn_hq_barriers(wcb, imageMemoryBarrierCount, pImageMemoryBarriers);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount,
                      const VkEvent *pEvents, VkPipelineStageFlags srcStageMask,
                      VkPipelineStageFlags dstStageMask,
                      uint32_t memoryBarrierCount,
                      const VkMemoryBarrier *pMemoryBarriers,
                      uint32_t bufferMemoryBarrierCount,
                      const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                      uint32_t imageMemoryBarrierCount,
                      const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkImageMemoryBarrier stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageMemoryBarrier *capped = NULL;

   if (imageMemoryBarrierCount &&
       __atomic_load_n(&wrapper_capped_images, __ATOMIC_RELAXED) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), imageMemoryBarrierCount))) {
      imageMemoryBarrierCount = wrapper_cap_image_barriers(device,
         pImageMemoryBarriers, imageMemoryBarrierCount, capped);
      pImageMemoryBarriers = capped;
   }

   device->dispatch_table.CmdWaitEvents(wcb->dispatch_handle, eventCount,
      pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers,
      bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount,
      pImageMemoryBarriers);
   wrapper_bcn_hq_barriers(wcb, imageMemoryBarrierCount, pImageMemoryBarriers);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                       const VkEvent *pEvents,
                       const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkDependencyInfo *deps = NULL;
   VkImageMemoryBarrier2 *capped = NULL;

   if (eventCount && __atomic_load_n(&wrapper_capped_images, __ATOMIC_RELAXED)) {
      uint32_t total = 0;
      for (uint32_t i = 0; i < eventCount; i++)
         total += pDependencyInfos[i].imageMemoryBarrierCount;
      deps = malloc(sizeof(*deps) * eventCount);
      capped = malloc(sizeof(*capped) * (total ? total : 1));
      if (deps && capped) {
         VkImageMemoryBarrier2 *next = capped;
         for (uint32_t i = 0; i < eventCount; i++) {
            deps[i] = pDependencyInfos[i];
            deps[i].imageMemoryBarrierCount = wrapper_cap_image_barriers2(device,
               pDependencyInfos[i].pImageMemoryBarriers,
               pDependencyInfos[i].imageMemoryBarrierCount, next);
            deps[i].pImageMemoryBarriers = next;
            next += deps[i].imageMemoryBarrierCount;
         }
         pDependencyInfos = deps;
      }
   }

   device->dispatch_table.CmdWaitEvents2(wcb->dispatch_handle, eventCount,
      pEvents, pDependencyInfos);
   for (uint32_t i = 0; i < eventCount; i++)
      wrapper_bcn_hq_barriers2(wcb, &pDependencyInfos[i]);
   free(deps);
   free(capped);
}

/* Regions of a copy with an emulated BC image on either side that the BCn
 * emulation recorded itself; keep[i] is set for the ones left to the driver.
 * Returns how many are left. */
static uint32_t
wrapper_bcn_copy_image(struct wrapper_command_buffer *wcb, VkImage srcImage,
                       VkImageLayout srcLayout, VkImage dstImage,
                       VkImageLayout dstLayout, uint32_t count,
                       const VkImageCopy *regions, bool *keep)
{
   struct wrapper_device *device = wcb->device;
   for (uint32_t i = 0; i < count; i++)
      keep[i] = true;
   struct wrapper_image *src = get_wrapper_image_from_handle(device, srcImage);
   struct wrapper_image *dst = get_wrapper_image_from_handle(device, dstImage);
   if (!src || !dst || src->mip_drop || dst->mip_drop)
      return count;
   bool src_emu = is_emulated_bcn(device->physical, src->info.format);
   bool dst_emu = is_emulated_bcn(device->physical, dst->info.format);
   if (!src_emu && !dst_emu)
      return count;
   uint32_t left = 0;
   for (uint32_t i = 0; i < count; i++) {
      keep[i] = !wrapper_bcn_gpu_copy_image_region(wcb, src, src_emu, srcLayout,
                                                   dst, dst_emu, dstLayout, &regions[i]);
      left += keep[i];
   }
   return left;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyImage(VkCommandBuffer commandBuffer,
                     VkImage srcImage, VkImageLayout srcImageLayout,
                     VkImage dstImage, VkImageLayout dstImageLayout,
                     uint32_t regionCount, const VkImageCopy *pRegions)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   uint32_t src_drop = wrapper_image_mip_drop(device, srcImage);
   uint32_t dst_drop = wrapper_image_mip_drop(device, dstImage);
   VkImageCopy stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageCopy *capped = NULL;

   if ((src_drop || dst_drop) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), regionCount))) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < regionCount; i++) {
         capped[n] = pRegions[i];
         if (wrapper_cap_level(src_drop, &capped[n].srcSubresource.mipLevel) &&
             wrapper_cap_level(dst_drop, &capped[n].dstSubresource.mipLevel))
            n++;
      }
      if (!n) {
         wrapper_cap_free(capped, stack);
         return;
      }
      regionCount = n;
      pRegions = capped;
   }

   if (!src_drop && !dst_drop && device->physical->emulate_bcn > 1 && regionCount) {
      bool keep[regionCount];
      uint32_t left = wrapper_bcn_copy_image(wcb, srcImage, srcImageLayout, dstImage,
         dstImageLayout, regionCount, pRegions, keep);
      if (left < regionCount) {
         VkImageCopy rest[left ? left : 1];
         uint32_t n = 0;
         for (uint32_t i = 0; i < regionCount; i++)
            if (keep[i])
               rest[n++] = pRegions[i];
         if (n)
            device->dispatch_table.CmdCopyImage(wcb->dispatch_handle, srcImage,
               srcImageLayout, dstImage, dstImageLayout, n, rest);
         return;
      }
   }

   device->dispatch_table.CmdCopyImage(wcb->dispatch_handle, srcImage,
      srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyImage2(VkCommandBuffer commandBuffer,
                      const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   uint32_t src_drop = wrapper_image_mip_drop(device, pCopyImageInfo->srcImage);
   uint32_t dst_drop = wrapper_image_mip_drop(device, pCopyImageInfo->dstImage);
   uint32_t count = pCopyImageInfo->regionCount;
   VkCopyImageInfo2 capped_info;
   VkImageCopy2 stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageCopy2 *capped = NULL;

   if ((src_drop || dst_drop) &&
       (capped = wrapper_cap_alloc(stack, sizeof(*capped), count))) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < count; i++) {
         capped[n] = pCopyImageInfo->pRegions[i];
         if (wrapper_cap_level(src_drop, &capped[n].srcSubresource.mipLevel) &&
             wrapper_cap_level(dst_drop, &capped[n].dstSubresource.mipLevel))
            n++;
      }
      if (!n) {
         wrapper_cap_free(capped, stack);
         return;
      }
      capped_info = *pCopyImageInfo;
      capped_info.regionCount = n;
      capped_info.pRegions = capped;
      pCopyImageInfo = &capped_info;
   }

   if (!src_drop && !dst_drop && device->physical->emulate_bcn > 1 && count) {
      VkImageCopy regions[count];
      bool keep[count];
      for (uint32_t i = 0; i < count; i++) {
         const VkImageCopy2 *r = &pCopyImageInfo->pRegions[i];
         regions[i] = (VkImageCopy){
            .srcSubresource = r->srcSubresource,
            .srcOffset = r->srcOffset,
            .dstSubresource = r->dstSubresource,
            .dstOffset = r->dstOffset,
            .extent = r->extent,
         };
      }
      uint32_t left = wrapper_bcn_copy_image(wcb, pCopyImageInfo->srcImage,
         pCopyImageInfo->srcImageLayout, pCopyImageInfo->dstImage,
         pCopyImageInfo->dstImageLayout, count, regions, keep);
      if (left < count) {
         VkImageCopy2 rest[left ? left : 1];
         uint32_t n = 0;
         for (uint32_t i = 0; i < count; i++)
            if (keep[i])
               rest[n++] = pCopyImageInfo->pRegions[i];
         if (n) {
            VkCopyImageInfo2 info = *pCopyImageInfo;
            info.regionCount = n;
            info.pRegions = rest;
            device->dispatch_table.CmdCopyImage2(wcb->dispatch_handle, &info);
         }
         return;
      }
   }

   device->dispatch_table.CmdCopyImage2(wcb->dispatch_handle, pCopyImageInfo);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image,
                           VkImageLayout imageLayout,
                           const VkClearColorValue *pColor,
                           uint32_t rangeCount,
                           const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   uint32_t mip_drop = wrapper_image_mip_drop(device, image);
   VkImageSubresourceRange stack[WRAPPER_CAP_STACK_ENTRIES];
   VkImageSubresourceRange *capped = NULL;

   if (mip_drop && (capped = wrapper_cap_alloc(stack, sizeof(*capped), rangeCount))) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < rangeCount; i++) {
         capped[n] = pRanges[i];
         if (bcn_cap_range(mip_drop, &capped[n]))
            n++;
      }
      if (!n) {
         wrapper_cap_free(capped, stack);
         return;
      }
      rangeCount = n;
      pRanges = capped;
   }

   device->dispatch_table.CmdClearColorImage(wcb->dispatch_handle, image,
      imageLayout, pColor, rangeCount, pRanges);
   wrapper_cap_free(capped, stack);
}

/* Readback of a dropped level returns the reduced base level (best effort),
 * with the region scaled to fit it and tightly packed. */
static void
wrapper_cap_readback_region(uint32_t mip_drop, VkImageSubresourceLayers *sub,
                            VkOffset3D *offset, VkExtent3D *extent,
                            uint32_t *row_length, uint32_t *image_height)
{
   if (wrapper_cap_level(mip_drop, &sub->mipLevel))
      return;
   uint32_t d = mip_drop - sub->mipLevel;
   sub->mipLevel = 0;
   offset->x >>= d;
   offset->y >>= d;
   extent->width = MAX2(1, extent->width >> d);
   extent->height = MAX2(1, extent->height >> d);
   *row_length = 0;
   *image_height = 0;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage,
                             VkImageLayout srcImageLayout, VkBuffer dstBuffer,
                             uint32_t regionCount,
                             const VkBufferImageCopy *pRegions)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   uint32_t mip_drop = wrapper_image_mip_drop(device, srcImage);
   VkBufferImageCopy stack[WRAPPER_CAP_STACK_ENTRIES];
   VkBufferImageCopy *capped = NULL;

   if (!mip_drop && device->physical->emulate_bcn > 1) {
      struct wrapper_image *wi = get_wrapper_image_from_handle(device, srcImage);
      if (wi && is_emulated_bcn(device->physical, wi->info.format) &&
          wrapper_bcn_gpu_readback(wcb, wi, srcImageLayout, dstBuffer, regionCount, pRegions))
         return;
   }

   if (mip_drop && (capped = wrapper_cap_alloc(stack, sizeof(*capped), regionCount))) {
      for (uint32_t i = 0; i < regionCount; i++) {
         capped[i] = pRegions[i];
         wrapper_cap_readback_region(mip_drop, &capped[i].imageSubresource,
            &capped[i].imageOffset, &capped[i].imageExtent,
            &capped[i].bufferRowLength, &capped[i].bufferImageHeight);
      }
      pRegions = capped;
   }

   device->dispatch_table.CmdCopyImageToBuffer(wcb->dispatch_handle, srcImage,
      srcImageLayout, dstBuffer, regionCount, pRegions);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer,
                              const VkCopyImageToBufferInfo2 *pInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   uint32_t mip_drop = wrapper_image_mip_drop(device, pInfo->srcImage);
   VkCopyImageToBufferInfo2 capped_info;
   VkBufferImageCopy2 stack[WRAPPER_CAP_STACK_ENTRIES];
   VkBufferImageCopy2 *capped = NULL;

   if (!mip_drop && device->physical->emulate_bcn > 1 && pInfo->regionCount) {
      struct wrapper_image *wi = get_wrapper_image_from_handle(device, pInfo->srcImage);
      if (wi && is_emulated_bcn(device->physical, wi->info.format)) {
         VkBufferImageCopy regions[pInfo->regionCount];
         for (uint32_t i = 0; i < pInfo->regionCount; i++) {
            const VkBufferImageCopy2 *r = &pInfo->pRegions[i];
            regions[i] = (VkBufferImageCopy){
               .bufferOffset = r->bufferOffset,
               .bufferRowLength = r->bufferRowLength,
               .bufferImageHeight = r->bufferImageHeight,
               .imageSubresource = r->imageSubresource,
               .imageOffset = r->imageOffset,
               .imageExtent = r->imageExtent,
            };
         }
         if (wrapper_bcn_gpu_readback(wcb, wi, pInfo->srcImageLayout, pInfo->dstBuffer,
                                      pInfo->regionCount, regions))
            return;
      }
   }

   if (mip_drop && (capped = wrapper_cap_alloc(stack, sizeof(*capped), pInfo->regionCount))) {
      for (uint32_t i = 0; i < pInfo->regionCount; i++) {
         capped[i] = pInfo->pRegions[i];
         wrapper_cap_readback_region(mip_drop, &capped[i].imageSubresource,
            &capped[i].imageOffset, &capped[i].imageExtent,
            &capped[i].bufferRowLength, &capped[i].bufferImageHeight);
      }
      capped_info = *pInfo;
      capped_info.pRegions = capped;
      pInfo = &capped_info;
   }

   device->dispatch_table.CmdCopyImageToBuffer2(wcb->dispatch_handle, pInfo);
   wrapper_cap_free(capped, stack);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetImageSubresourceLayout(VkDevice _device, VkImage image,
                                  const VkImageSubresource *pSubresource,
                                  VkSubresourceLayout *pLayout)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkImageSubresource capped_sub;
   uint32_t mip_drop = wrapper_image_mip_drop(device, image);

   if (mip_drop) {
      capped_sub = *pSubresource;
      if (!wrapper_cap_level(mip_drop, &capped_sub.mipLevel))
         capped_sub.mipLevel = 0;
      pSubresource = &capped_sub;
   }

   device->dispatch_table.GetImageSubresourceLayout(device->dispatch_handle,
      image, pSubresource, pLayout);

   if (wrapper_diag_on()) {
      struct wrapper_image *wi = get_wrapper_image_from_handle(device, image);
      if (wi && is_emulated_bcn(device->physical, wi->info.format))
         wrapper_diag_append(
            "[FMT] SubresourceLayout fmt=%d mip=%u layer=%u off=%llu size=%llu row=%llu\n",
            wi->info.format, pSubresource->mipLevel, pSubresource->arrayLayer,
            (unsigned long long)pLayout->offset, (unsigned long long)pLayout->size,
            (unsigned long long)pLayout->rowPitch);
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_FreeCommandBuffers(VkDevice _device,
                           VkCommandPool commandPool,
                           uint32_t commandBufferCount,
                           const VkCommandBuffer* pCommandBuffers)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   simple_mtx_lock(&device->resource_mutex);

   for (int i = 0; i < commandBufferCount; i++) {
      VK_FROM_HANDLE(wrapper_command_buffer, wcb, pCommandBuffers[i]);
      wrapper_command_buffer_destroy(device, wcb);
   }

   simple_mtx_unlock(&device->resource_mutex);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyCommandPool(VkDevice _device, VkCommandPool commandPool,
                           const VkAllocationCallbacks* pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   simple_mtx_lock(&device->resource_mutex);

   list_for_each_entry_safe(struct wrapper_command_buffer, wcb,
                            &device->command_buffer_list, link) {
      if (wcb->pool == commandPool) {
         wrapper_command_buffer_destroy(device, wcb);
      }
   }

   simple_mtx_unlock(&device->resource_mutex);

   device->dispatch_table.DestroyCommandPool(device->dispatch_handle,
                                             commandPool, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_DestroyDevice(VkDevice _device, const VkAllocationCallbacks* pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);

   wrapper_query_reset_finish(device);

   simple_mtx_lock(&device->resource_mutex);

   list_for_each_entry_safe(struct wrapper_command_buffer, wcb,
                            &device->command_buffer_list, link) {
      wrapper_command_buffer_destroy(device, wcb);
   }
   list_for_each_entry_safe(struct wrapper_device_memory, mem,
                            &device->device_memory_list, link) {
      wrapper_device_memory_destroy(mem);
   }
   
   simple_mtx_unlock(&device->resource_mutex);
   
   list_for_each_entry_safe(struct wrapper_buffer, wb,
                            &device->buffer_list, link) {
      wrapper_buffer_destroy(device, wb, pAllocator);
   }
   list_for_each_entry_safe(struct wrapper_image, wi,
                            &device->image_list, link) {
      wrapper_image_destroy(device, wi, pAllocator);
   }
   list_for_each_entry_safe(struct wrapper_fence, wf,
                            &device->fence_list, link) {
      wrapper_fence_destroy(device, wf, pAllocator);
   }

   hash_table_u64_foreach(device->dynamic_pipeline_table, entry) {
      struct wrapper_dynamic_pipeline *pipeline = entry.data;
      if (pipeline->render_pass)
         device->dispatch_table.DestroyRenderPass(device->dispatch_handle,
                                                   pipeline->render_pass, NULL);
      free(pipeline);
   }
   hash_table_u64_foreach(device->image_view_table, entry)
      free(entry.data);
   hash_table_u64_foreach(device->imageless_fb_table, entry)
      free(entry.data);
   _mesa_hash_table_u64_destroy(device->dynamic_pipeline_table);
   wrapper_stride_finish(device);
   wrapper_divisor_finish(device);
   _mesa_hash_table_u64_destroy(device->image_view_table);
   _mesa_hash_table_u64_destroy(device->imageless_fb_table);

   list_for_each_entry_safe(struct vk_queue, queue, &device->vk.queues, link) {
      vk_queue_finish(queue);
      vk_free2(&device->vk.alloc, pAllocator, queue);
   }
   if (device->dispatch_handle != VK_NULL_HANDLE) {
      wrapper_bcn_hq_finish(device);
      wrapper_bcn_cache_report();
      wrapper_bcn_gpu_finish(device);
      device->dispatch_table.DestroyDevice(device->
         dispatch_handle, pAllocator);
   }
   if (device->emulate_push_descriptor) {
      hash_table_u64_foreach(device->push_dsl_table, e)
         free(e.data);
      hash_table_u64_foreach(device->push_pl_table, e) {
         struct wrapper_push_pl *r = e.data;
         free(r->set_layouts);
         free(r);
      }
      hash_table_u64_foreach(device->push_template_table, e)
         free(e.data);
      _mesa_hash_table_u64_destroy(device->push_dsl_table);
      _mesa_hash_table_u64_destroy(device->push_pl_table);
      _mesa_hash_table_u64_destroy(device->push_template_table);
      simple_mtx_destroy(&device->push_mutex);
   }
   simple_mtx_destroy(&device->resource_mutex);
   simple_mtx_destroy(&device->query_reset_mutex);
   vk_device_finish(&device->vk);
   vk_free2(&device->vk.alloc, pAllocator, device);
}

static uint64_t
unwrap_device_object(VkObjectType objectType,
                     uint64_t objectHandle)
{
   switch(objectType) {
   case VK_OBJECT_TYPE_DEVICE:
      return (uint64_t)(uintptr_t)wrapper_device_from_handle((VkDevice)(uintptr_t)objectHandle)->dispatch_handle;
   case VK_OBJECT_TYPE_QUEUE:
      return (uint64_t)(uintptr_t)wrapper_queue_from_handle((VkQueue)(uintptr_t)objectHandle)->dispatch_handle;
   case VK_OBJECT_TYPE_COMMAND_BUFFER:
      return (uint64_t)(uintptr_t)wrapper_command_buffer_from_handle((VkCommandBuffer)(uintptr_t)objectHandle)->dispatch_handle;
   default:
      return objectHandle;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_SetPrivateData(VkDevice _device, VkObjectType objectType,
                       uint64_t objectHandle,
                       VkPrivateDataSlot privateDataSlot,
                       uint64_t data) {
   VK_FROM_HANDLE(wrapper_device, device, _device);

   uint64_t object_handle = unwrap_device_object(objectType, objectHandle);
   return device->dispatch_table.SetPrivateData(device->dispatch_handle,
      objectType, object_handle, privateDataSlot, data);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPrivateData(VkDevice _device, VkObjectType objectType,
                       uint64_t objectHandle,
                       VkPrivateDataSlot privateDataSlot,
                       uint64_t* pData) {
   VK_FROM_HANDLE(wrapper_device, device, _device);

   uint64_t object_handle = unwrap_device_object(objectType, objectHandle);
   return device->dispatch_table.GetPrivateData(device->dispatch_handle,
      objectType, object_handle, privateDataSlot, pData);
}

static PFN_vkCmdCopyBuffer legacy_copy_buffer;
static PFN_vkCmdCopyImage legacy_copy_image;
static PFN_vkCmdBlitImage legacy_blit_image;
static PFN_vkCmdCopyBufferToImage legacy_copy_buffer_to_image;
static PFN_vkCmdCopyImageToBuffer legacy_copy_image_to_buffer;
static PFN_vkCmdResolveImage legacy_resolve_image;
static PFN_vkCmdWriteTimestamp legacy_write_timestamp;
static PFN_vkCmdSetEvent legacy_set_event;
static PFN_vkCmdResetEvent legacy_reset_event;
static PFN_vkCmdWaitEvents legacy_wait_events;

static VKAPI_ATTR void VKAPI_CALL
lower_copy_buffer2(VkCommandBuffer cb, const VkCopyBufferInfo2 *info)
{
   VkBufferCopy *r = malloc(MAX2(info->regionCount, 1) * sizeof(*r));
   if (!r)
      return;
   for (uint32_t i = 0; i < info->regionCount; i++)
      r[i] = (VkBufferCopy) { info->pRegions[i].srcOffset, info->pRegions[i].dstOffset,
                              info->pRegions[i].size };
   legacy_copy_buffer(cb, info->srcBuffer, info->dstBuffer, info->regionCount, r);
   free(r);
}

static VKAPI_ATTR void VKAPI_CALL
lower_copy_image2(VkCommandBuffer cb, const VkCopyImageInfo2 *info)
{
   VkImageCopy *r = malloc(MAX2(info->regionCount, 1) * sizeof(*r));
   if (!r)
      return;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *s = &info->pRegions[i];
      r[i] = (VkImageCopy) { s->srcSubresource, s->srcOffset, s->dstSubresource,
                             s->dstOffset, s->extent };
   }
   legacy_copy_image(cb, info->srcImage, info->srcImageLayout, info->dstImage,
                     info->dstImageLayout, info->regionCount, r);
   free(r);
}

static VKAPI_ATTR void VKAPI_CALL
lower_blit_image2(VkCommandBuffer cb, const VkBlitImageInfo2 *info)
{
   VkImageBlit *r = malloc(MAX2(info->regionCount, 1) * sizeof(*r));
   if (!r)
      return;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageBlit2 *s = &info->pRegions[i];
      r[i].srcSubresource = s->srcSubresource;
      r[i].srcOffsets[0] = s->srcOffsets[0];
      r[i].srcOffsets[1] = s->srcOffsets[1];
      r[i].dstSubresource = s->dstSubresource;
      r[i].dstOffsets[0] = s->dstOffsets[0];
      r[i].dstOffsets[1] = s->dstOffsets[1];
   }
   legacy_blit_image(cb, info->srcImage, info->srcImageLayout, info->dstImage,
                     info->dstImageLayout, info->regionCount, r, info->filter);
   free(r);
}

static VkBufferImageCopy *
lower_buffer_image_regions(const VkBufferImageCopy2 *in, uint32_t count)
{
   VkBufferImageCopy *r = malloc(MAX2(count, 1) * sizeof(*r));
   if (!r)
      return NULL;
   for (uint32_t i = 0; i < count; i++)
      r[i] = (VkBufferImageCopy) { in[i].bufferOffset, in[i].bufferRowLength,
                                   in[i].bufferImageHeight, in[i].imageSubresource,
                                   in[i].imageOffset, in[i].imageExtent };
   return r;
}

static VKAPI_ATTR void VKAPI_CALL
lower_copy_buffer_to_image2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2 *info)
{
   VkBufferImageCopy *r = lower_buffer_image_regions(info->pRegions, info->regionCount);
   if (!r)
      return;
   legacy_copy_buffer_to_image(cb, info->srcBuffer, info->dstImage, info->dstImageLayout,
                               info->regionCount, r);
   free(r);
}

static VKAPI_ATTR void VKAPI_CALL
lower_copy_image_to_buffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2 *info)
{
   VkBufferImageCopy *r = lower_buffer_image_regions(info->pRegions, info->regionCount);
   if (!r)
      return;
   legacy_copy_image_to_buffer(cb, info->srcImage, info->srcImageLayout, info->dstBuffer,
                               info->regionCount, r);
   free(r);
}

static VKAPI_ATTR void VKAPI_CALL
lower_resolve_image2(VkCommandBuffer cb, const VkResolveImageInfo2 *info)
{
   VkImageResolve *r = malloc(MAX2(info->regionCount, 1) * sizeof(*r));
   if (!r)
      return;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageResolve2 *s = &info->pRegions[i];
      r[i] = (VkImageResolve) { s->srcSubresource, s->srcOffset, s->dstSubresource,
                                s->dstOffset, s->extent };
   }
   legacy_resolve_image(cb, info->srcImage, info->srcImageLayout, info->dstImage,
                        info->dstImageLayout, info->regionCount, r);
   free(r);
}

static VKAPI_ATTR void VKAPI_CALL
lower_write_timestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 stage,
                       VkQueryPool pool, uint32_t query)
{
   legacy_write_timestamp(cb, (VkPipelineStageFlagBits)wrapper_stage_mask2_to_legacy(stage, false),
                          pool, query);
}

static VkPipelineStageFlags
dependency_src_stages(const VkDependencyInfo *dep)
{
   VkPipelineStageFlags2 stages = 0;
   for (uint32_t i = 0; i < dep->memoryBarrierCount; i++)
      stages |= dep->pMemoryBarriers[i].srcStageMask;
   for (uint32_t i = 0; i < dep->bufferMemoryBarrierCount; i++)
      stages |= dep->pBufferMemoryBarriers[i].srcStageMask;
   for (uint32_t i = 0; i < dep->imageMemoryBarrierCount; i++)
      stages |= dep->pImageMemoryBarriers[i].srcStageMask;
   return wrapper_stage_mask2_to_legacy(stages, true);
}

static VKAPI_ATTR void VKAPI_CALL
lower_set_event2(VkCommandBuffer cb, VkEvent event, const VkDependencyInfo *dep)
{
   legacy_set_event(cb, event, dependency_src_stages(dep));
}

static VKAPI_ATTR void VKAPI_CALL
lower_reset_event2(VkCommandBuffer cb, VkEvent event, VkPipelineStageFlags2 stage)
{
   legacy_reset_event(cb, event, wrapper_stage_mask2_to_legacy(stage, true));
}

static VKAPI_ATTR void VKAPI_CALL
lower_wait_events2(VkCommandBuffer cb, uint32_t count, const VkEvent *events,
                   const VkDependencyInfo *deps)
{
   VkPipelineStageFlags src = 0;
   for (uint32_t i = 0; i < count; i++)
      src |= dependency_src_stages(&deps[i]);
   VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   legacy_wait_events(cb, count, events, src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      1, &barrier, 0, NULL, 0, NULL);
}

static void
wrapper_lower_missing_commands(struct wrapper_device *device)
{
   struct vk_device_dispatch_table *t = &device->dispatch_table;

   legacy_copy_buffer = t->CmdCopyBuffer;
   legacy_copy_image = t->CmdCopyImage;
   legacy_blit_image = t->CmdBlitImage;
   legacy_copy_buffer_to_image = t->CmdCopyBufferToImage;
   legacy_copy_image_to_buffer = t->CmdCopyImageToBuffer;
   legacy_resolve_image = t->CmdResolveImage;
   legacy_write_timestamp = t->CmdWriteTimestamp;
   legacy_set_event = t->CmdSetEvent;
   legacy_reset_event = t->CmdResetEvent;
   legacy_wait_events = t->CmdWaitEvents;

#define LOWER(name, fn) \
   if (!t->name) { \
      t->name = fn; \
      WRAPPER_LOG(info, "Lowering vk" #name " to the legacy command"); \
   }
   LOWER(CmdCopyBuffer2, lower_copy_buffer2);
   LOWER(CmdCopyImage2, lower_copy_image2);
   LOWER(CmdBlitImage2, lower_blit_image2);
   LOWER(CmdCopyBufferToImage2, lower_copy_buffer_to_image2);
   LOWER(CmdCopyImageToBuffer2, lower_copy_image_to_buffer2);
   LOWER(CmdResolveImage2, lower_resolve_image2);
   LOWER(CmdWriteTimestamp2, lower_write_timestamp2);
   LOWER(CmdSetEvent2, lower_set_event2);
   LOWER(CmdResetEvent2, lower_reset_event2);
   LOWER(CmdWaitEvents2, lower_wait_events2);
#undef LOWER
}
