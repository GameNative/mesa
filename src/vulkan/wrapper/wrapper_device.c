#include <sys/stat.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_bcdec.h"
#include "wrapper_bcn_spv.h"
#include "spirv_patcher.hpp"
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
               VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
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
   if (device->emulate_null_descriptor && pBuffers) {
      bufs = malloc(sizeof(VkBuffer) * bindingCount);
      for (uint32_t i = 0; i < bindingCount; i++)
         bufs[i] = pBuffers[i] ? pBuffers[i] : device->null_buffer;
   }
   if (device->dispatch_table.CmdBindVertexBuffers2)
      device->dispatch_table.CmdBindVertexBuffers2(wcb->dispatch_handle,
         firstBinding, bindingCount, bufs ? bufs : pBuffers, pOffsets, pSizes,
         pStrides);
   else if (wrapper_lowers_core13(device) &&
            device->dispatch_table.CmdBindVertexBuffers2EXT)
      device->dispatch_table.CmdBindVertexBuffers2EXT(wcb->dispatch_handle,
         firstBinding, bindingCount, bufs ? bufs : pBuffers, pOffsets, pSizes,
         pStrides);
   free(bufs);
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
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                            uint32_t bindingCount, const VkBuffer *pBuffers,
                            const VkDeviceSize *pOffsets)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   struct wrapper_device *device = wcb->device;
   VkBuffer *bufs = NULL;
   if (device->emulate_null_descriptor && pBuffers) {
      bufs = malloc(sizeof(VkBuffer) * bindingCount);
      for (uint32_t i = 0; i < bindingCount; i++)
         bufs[i] = pBuffers[i] ? pBuffers[i] : device->null_buffer;
   }
   device->dispatch_table.CmdBindVertexBuffers(wcb->dispatch_handle, firstBinding,
      bindingCount, bufs ? bufs : pBuffers, pOffsets);
   free(bufs);
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
      if (!redirected && e && (strstr(e, "DXVK") || strstr(e, "vkd3d"))) {
         redirected = 1;
         int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
         if (fd >= 0) {
            dup2(fd, 1);   /* stdout: vkd3d/wine 'System.out' errors */
            dup2(fd, 2);   /* stderr */
            if (fd > 2) close(fd);
            setvbuf(stdout, NULL, _IONBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
         }
      }
   }

#define D(...) fprintf(stderr, "[WRAPPER_DIAG] " __VA_ARGS__)

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
   D("  BCn: emulate=%d  ASTC=%s  transcode=%s  cache=%s\n",
     pdev->emulate_bcn,
     getenv("WRAPPER_ASTC_BLOCK") ? getenv("WRAPPER_ASTC_BLOCK") : "4x4",
     (getenv("WRAPPER_BCN_GPU") && atoi(getenv("WRAPPER_BCN_GPU"))) ? "GPU" : "CPU",
     (getenv("WRAPPER_USE_BCN_CACHE") && atoi(getenv("WRAPPER_USE_BCN_CACHE"))) ? "on" : "off");
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
   device->buffer_table = _mesa_hash_table_u64_create(NULL);
   device->fence_table = _mesa_hash_table_u64_create(NULL);
   
   simple_mtx_init(&device->resource_mutex, mtx_plain);
   simple_mtx_init(&device->bcn_gpu_mutex, mtx_plain);
   simple_mtx_init(&device->query_reset_mutex, mtx_plain);
   device->bcn_gpu_state = 0;
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

#undef DISABLE_FEATURE

   process_pnext_chain((VkBaseInStructure *)&wrapper_create_info, device->physical);

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
      wrapper_safe_create_device = getenv("WRAPPER_SAFE_CREATE_DEVICE") ? atoi(getenv("WRAPPER_SAFE_CREATE_DEVICE")) : 1;
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

   result = physical_device->dispatch_table.CreateDevice(
      physical_device->dispatch_handle, &wrapper_create_info,
         pAllocator, &device->dispatch_handle);

   if (result != VK_SUCCESS) {
      if (wrapper_safe_create_device) {
         WRAPPER_LOG(info, "Forcing device creation with a NULL pNext chain");
         wrapper_create_info.pNext = NULL;
         used_fallback_create = true;
         result = physical_device->dispatch_table.CreateDevice(
            physical_device->dispatch_handle, &wrapper_create_info,
               pAllocator, &device->dispatch_handle);
      }
      
      if (result != VK_SUCCESS) {
         free(query_reset_queues);
         free(query_reset_priorities);
         WRAPPER_LOG(error, "Failed driver createDevice, res %d", result);
         wrapper_emit_diag(physical_device, pCreateInfo, result);
         wrapper_DestroyDevice(wrapper_device_to_handle(device),
                               &device->vk.alloc);
         return vk_error(physical_device, result);
      }
   }

   free(query_reset_queues);
   free(query_reset_priorities);

   void *gdpa = physical_device->instance->dispatch_table.GetInstanceProcAddr(
      physical_device->instance->dispatch_handle, "vkGetDeviceProcAddr");
   vk_device_dispatch_table_load(&device->dispatch_table, gdpa,
                                 device->dispatch_handle);

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

   simple_mtx_unlock(&device->resource_mutex);
   
   vk_object_free(&device->vk, &device->vk.alloc, wi);
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
      create_info.format = get_format_for_bcn(pCreateInfo->format);
      if (create_info.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)
         create_info.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

      for (const VkBaseInStructure *s = pCreateInfo->pNext; s; s = s->pNext) {
         if (s->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
            const VkImageFormatListCreateInfo *fl =
               (const VkImageFormatListCreateInfo *)s;
            for (uint32_t i = 0; i < fl->viewFormatCount; i++) {
               if (is_emulated_bcn(device->physical, fl->pViewFormats[i]))
                  ((VkFormat *)fl->pViewFormats)[i] =
                     get_format_for_bcn(fl->pViewFormats[i]);
            }
         }
      }
   }

   res = device->dispatch_table.CreateImage(device->dispatch_handle,
      &create_info, pAllocator, pImage);

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

   list_add(&wi->link, &device->image_list);
   _mesa_hash_table_u64_insert(device->image_table, (uint64_t)wi->dispatch_handle, wi);

   simple_mtx_unlock(&device->resource_mutex);

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
      ci.format = get_format_for_bcn(pInfo->pCreateInfo->format);
      ci.flags &= ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      ci.pNext = NULL;
      info.pCreateInfo = &ci;
   }

   device->dispatch_table.GetDeviceImageMemoryRequirements(device->dispatch_handle,
      &info, pMemoryRequirements);
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
   return vk_device_get_proc_addr(&device->vk, pName);
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

   dt->CmdBindDescriptorSets(wcb->dispatch_handle, bindPoint, layout, set,
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
   dt->CmdBindDescriptorSets(wcb->dispatch_handle, bp, layout, set,
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

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateDescriptorSetLayout(VkDevice _device,
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

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreatePipelineLayout(VkDevice _device,
                            const VkPipelineLayoutCreateInfo *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult r = device->dispatch_table.CreatePipelineLayout(
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
   device->dispatch_table.DestroyPipelineLayout(device->dispatch_handle,
      pipelineLayout, pAllocator);
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

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                          const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wrapper_dynamic_render_objects_reset(wcb);
   if (wcb->device->emulate_push_descriptor)
      wrapper_push_pool_reset_all(wcb);
   VkResult begin_result =
      wcb->device->dispatch_table.BeginCommandBuffer(wcb->dispatch_handle,
                                                     pBeginInfo);
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

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                            const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
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
   const VkGraphicsPipelineCreateInfo *driver_infos = pCreateInfos;

   if (wrapper_lowers_core13(device) &&
       !device->dispatch_table.CmdBeginRendering && createInfoCount) {
      legacy = calloc(createInfoCount, sizeof(*legacy));
      render_passes = calloc(createInfoCount, sizeof(*render_passes));
      if (!legacy || !render_passes) {
         free(legacy);
         free(render_passes);
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
            return rp_result;
         }
         legacy[i].renderPass = render_passes[i];
         legacy[i].subpass = 0;
         /* The base ICD does not support dynamic rendering.  Once a compatible
          * legacy render pass is supplied, do not also pass its Vulkan 1.3
          * VkPipelineRenderingCreateInfo to that Vulkan 1.1 driver.  D8VK puts
          * this structure at the head of the chain (and currently it is the
          * only entry), so unlink it from the driver-facing create info. */
         if (pCreateInfos[i].pNext == rendering)
            legacy[i].pNext = rendering->pNext;
         WRAPPER_LOG(info,
            "Lowering dynamic graphics pipeline %u to legacy render pass %p; strippedRenderingInfo=%u",
            i, (void *)(uintptr_t)render_passes[i],
            pCreateInfos[i].pNext == rendering);
      }
      driver_infos = legacy;
   }

   VkResult result = device->dispatch_table.CreateGraphicsPipelines(
      device->dispatch_handle, pipelineCache, createInfoCount, driver_infos,
      pAllocator, pPipelines);

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
   wcb->device->dispatch_table.CmdDraw(wcb->dispatch_handle, vertexCount,
                                       instanceCount, firstVertex, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                       uint32_t instanceCount, uint32_t firstIndex,
                       int32_t vertexOffset, uint32_t firstInstance)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wcb->device->dispatch_table.CmdDrawIndexed(
      wcb->dispatch_handle, indexCount, instanceCount, firstIndex,
      vertexOffset, firstInstance);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_ResetCommandBuffer(VkCommandBuffer commandBuffer,
                          VkCommandBufferResetFlags flags)
{
   VK_FROM_HANDLE(wrapper_command_buffer, wcb, commandBuffer);
   wrapper_dynamic_render_objects_reset(wcb);
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
       device->physical->emulate_vulkan13) {
      simple_mtx_lock(&device->resource_mutex);
      list_for_each_entry(struct wrapper_command_buffer, wcb,
                          &device->command_buffer_list, link) {
         if (wcb->pool == commandPool) {
            wrapper_dynamic_render_objects_reset(wcb);
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

   for (int i = 0; i < submitCount; i++) {
      const VkSubmitInfo *submit_info = &pSubmits[i];
      command_buffers = malloc(sizeof(VkCommandBuffer) *
         submit_info->commandBufferCount);
      for (int j = 0; j < submit_info->commandBufferCount; j++) {
         VK_FROM_HANDLE(wrapper_command_buffer, wcb,
                        submit_info->pCommandBuffers[j]);
         wcb->fence = wf;
         command_buffers[j] = wcb->dispatch_handle;
         
      }
      wrapper_submits[i] = pSubmits[i];
      wrapper_submits[i].pCommandBuffers = command_buffers;
   }

   if (serialise)
      simple_mtx_lock(&queue->device->query_reset_mutex);
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
      }
      wrapper_submits[i] = pSubmits[i];
      wrapper_submits[i].pCommandBufferInfos = command_buffers;
   }

   if (serialise)
      simple_mtx_lock(&queue->device->query_reset_mutex);
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

   list_inithead(&wf->staging_buffers_list);

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

   if (res != VK_SUCCESS || device->physical->emulate_bcn < 2)
      return res;

   for (uint32_t i = 0; i < fenceCount; i++) {
      struct wrapper_fence *wf = get_wrapper_fence_from_handle(device, pFences[i]);
      list_for_each_entry_safe(struct wrapper_buffer, wb,
                               &wf->staging_buffers_list, link)
      {
         VkDeviceMemory memory = wb->memory;
         if (wb->desc_pool)
            device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle,
               wb->desc_pool, NULL);
         if (wb->bcn_inflight) {
            simple_mtx_lock(&device->bcn_gpu_mutex);
            device->bcn_gpu_inflight -= wb->bcn_inflight;
            simple_mtx_unlock(&device->bcn_gpu_mutex);
         }
         wrapper_buffer_destroy(device, wb, NULL);
         device->dispatch_table.FreeMemory(device->dispatch_handle,
            memory, NULL);
      }
   }

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

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateShaderModule(VkDevice _device,
						   const VkShaderModuleCreateInfo *pCreateInfo,
						   const VkAllocationCallbacks *pAllocator,
						   VkShaderModule *pShaderModule)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   static int wrapper_no_remove_clip_distance = -1;
   static int wrapper_no_patch_OpConstComp = -1;

   if (wrapper_no_remove_clip_distance == -1)
      wrapper_no_remove_clip_distance = getenv("WRAPPER_NO_REMOVE_CLIP_DISTANCE") && atoi(getenv("WRAPPER_NO_REMOVE_CLIP_DISTANCE"));

   if (wrapper_no_patch_OpConstComp == -1)
      wrapper_no_patch_OpConstComp = getenv("WRAPPER_NO_PATCH_OPCONSTCOMP") && atoi(getenv("WRAPPER_NO_PATCH_OPCONSTCOMP"));
      
   VkShaderModuleCreateInfo create_info = *pCreateInfo;

   simple_mtx_lock(&device->resource_mutex);

   if (device->physical->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY) {
      uint32_t *code = malloc(create_info.codeSize);
      memcpy(code, create_info.pCode, create_info.codeSize);
      if (!wrapper_no_patch_OpConstComp) patch_OpConstantComposite_to_OpSpecConstantComposite(code, create_info.codeSize);
      if (!wrapper_no_remove_clip_distance) remove_ClipDistance(code, &create_info.codeSize);
      create_info.pCode = code;
   }

   simple_mtx_unlock(&device->resource_mutex);

   if (WRAPPER_LOG_LEVEL(shader))
      dump_shader_code(create_info.pCode, create_info.codeSize);
   
   return device->dispatch_table.CreateShaderModule(
      device->dispatch_handle, &create_info, pAllocator, pShaderModule);
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

struct wrapper_bcn_pc {
   uint32_t block_x, block_x_src, block_y, src_word_off, format, has_alpha;
   uint32_t astc8, bc_bx, bc_by;
};

/* Shader format id for a BC format, or -1 if the GPU shader can't decode it
 * (those fall back to the CPU path). The shader now produces ASTC 4x4 OR 8x8
 * (selected per-dispatch via the astc8 push constant), so both are on the GPU. */
static int
wrapper_bcn_gpu_format_id(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      return 0; /* BC1: opaque, single-plane RGB */
   case VK_FORMAT_BC7_UNORM_BLOCK:
   case VK_FORMAT_BC7_SRGB_BLOCK:
      return 1; /* BC7: alpha, dual-plane */
   default:
      return -1;
   }
}

static void
wrapper_bcn_gpu_init(struct wrapper_device *device)
{
   VkResult r;
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(wrapper_bcn_transcode_spv),
      .pCode = wrapper_bcn_transcode_spv,
   };
   r = device->dispatch_table.CreateShaderModule(device->dispatch_handle,
      &smci, NULL, &device->bcn_shader);
   if (r != VK_SUCCESS) { device->bcn_gpu_state = -1; return; }

   VkDescriptorSetLayoutBinding binds[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = binds,
   };
   r = device->dispatch_table.CreateDescriptorSetLayout(device->dispatch_handle,
      &dslci, NULL, &device->bcn_set_layout);
   if (r != VK_SUCCESS) { device->bcn_gpu_state = -1; return; }

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0, .size = sizeof(struct wrapper_bcn_pc),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &device->bcn_set_layout,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   r = device->dispatch_table.CreatePipelineLayout(device->dispatch_handle,
      &plci, NULL, &device->bcn_pipe_layout);
   if (r != VK_SUCCESS) { device->bcn_gpu_state = -1; return; }

   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = device->bcn_shader, .pName = "main",
      },
      .layout = device->bcn_pipe_layout,
   };
   r = device->dispatch_table.CreateComputePipelines(device->dispatch_handle,
      VK_NULL_HANDLE, 1, &cpci, NULL, &device->bcn_pipeline);
   if (r != VK_SUCCESS) { device->bcn_gpu_state = -1; return; }

   device->bcn_gpu_state = 1;
}

static bool
wrapper_bcn_gpu_ready(struct wrapper_device *device)
{
   /* Default OFF. The GPU compute transcode is correct in isolation (its BC7
    * decode is bit-identical to bcdec and the ASTC encode matches the CPU
    * encoder), but running compute dispatches inline during Hades's launch
    * screen corrupts that screen's rendering (it renders black), and the boot
    * is package-load-bound so the GPU path saves no meaningful time. The CPU
    * encoder is correct everywhere; opt into the GPU path with WRAPPER_BCN_GPU=1. */
   static int env = -1;
   if (env == -1)
      env = getenv("WRAPPER_BCN_GPU") ? atoi(getenv("WRAPPER_BCN_GPU")) : 0;
   if (!env)
      return false;

   simple_mtx_lock(&device->bcn_gpu_mutex);
   if (device->bcn_gpu_state == 0)
      wrapper_bcn_gpu_init(device);
   bool ready = device->bcn_gpu_state == 1;
   simple_mtx_unlock(&device->bcn_gpu_mutex);
   return ready;
}

static struct wrapper_buffer *
wrapper_bcn_make_buffer(struct wrapper_device *device, VkDeviceSize size,
   VkBufferUsageFlags usage)
{
   struct wrapper_buffer *b = vk_object_zalloc(&device->vk, &device->vk.alloc,
      sizeof(struct wrapper_buffer), VK_OBJECT_TYPE_BUFFER);
   if (!b) return NULL;
   b->device = device;

   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &bci, NULL,
         &b->dispatch_handle) != VK_SUCCESS) {
      vk_object_free(&device->vk, &device->vk.alloc, b);
      return NULL;
   }
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(device->dispatch_handle,
      b->dispatch_handle, &mr);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = wrapper_select_device_memory_type(device,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   if (device->dispatch_table.AllocateMemory(device->dispatch_handle, &ai, NULL,
         &b->memory) != VK_SUCCESS) {
      device->dispatch_table.DestroyBuffer(device->dispatch_handle,
         b->dispatch_handle, NULL);
      vk_object_free(&device->vk, &device->vk.alloc, b);
      return NULL;
   }
   device->dispatch_table.BindBufferMemory(device->dispatch_handle,
      b->dispatch_handle, b->memory, 0);
   b->size = size;
   return b;
}

/* Destroy an untracked transcode buffer (error-path cleanup). */
static void
wrapper_bcn_free_buffer(struct wrapper_device *device, struct wrapper_buffer *b)
{
   if (!b) return;
   device->dispatch_table.DestroyBuffer(device->dispatch_handle, b->dispatch_handle, NULL);
   device->dispatch_table.FreeMemory(device->dispatch_handle, b->memory, NULL);
   vk_object_free(&device->vk, &device->vk.alloc, b);
}

/* GPU transcode: BC source buffer -> compute -> ASTC buffer -> copy to image.
 * Returns false (with nothing recorded for the failing region) to fall back to
 * the CPU path. Injected into the app's command buffer; state it binds (compute
 * pipeline/descriptors) is rebound by the app before its own next dispatch. */
static bool
wrapper_bcn_gpu_copy(struct wrapper_command_buffer *wcb,
                     struct wrapper_device *device, struct wrapper_buffer *wb,
                     VkImage dstImage, VkImageLayout dstLayout, VkFormat format,
                     int fmt_id, uint32_t regionCount,
                     const VkBufferImageCopy *pRegions)
{
   int block_size = (fmt_id == 0) ? 8 : 16;   /* BC1=8, BC7=16 bytes/block */
   uint32_t has_alpha = (fmt_id == 0) ? 0 : 1; /* BC1 opaque, BC7 alpha */
   int astc8 = is_astc_8x8(get_format_for_bcn(format)) ? 1 : 0;
   /* In-flight transient ceiling; beyond it, uploads fall back to CPU. Tunable
    * per device RAM via WRAPPER_BCN_GPU_CAP_MB (default 128 MB). */
   static VkDeviceSize CAP = 0;
   if (CAP == 0) {
      int mb = getenv("WRAPPER_BCN_GPU_CAP_MB") ? atoi(getenv("WRAPPER_BCN_GPU_CAP_MB")) : 128;
      if (mb < 16) mb = 16;
      CAP = (VkDeviceSize)mb * 1024 * 1024;
   }

   for (uint32_t i = 0; i < regionCount; i++) {
      VkBufferImageCopy region = pRegions[i];
      int w = region.imageExtent.width;
      int h = region.imageExtent.height;
      VkDeviceSize offset = region.bufferOffset;
      if (offset % 4 != 0)
         return false; /* CmdCopyBuffer needs 4-aligned offsets; fall back to CPU */
      int src_w = region.bufferRowLength ? (int)region.bufferRowLength : w;
      /* BC-block extents of the real texture, and the source row stride. */
      int bc_bx = (w + 3) / 4;
      int bc_by = (h + 3) / 4;
      int block_x_src = (src_w + 3) / 4;
      /* Destination ASTC block grid: 8x8 blocks pack 2x2 BC blocks each. */
      int block_x = astc8 ? (w + 7) / 8 : bc_bx;
      int block_y = astc8 ? (h + 7) / 8 : bc_by;
      VkDeviceSize src_size = (VkDeviceSize)block_x_src * bc_by * block_size;
      VkDeviceSize dst_size = (VkDeviceSize)block_x * block_y * 16;
      VkDeviceSize total = src_size + dst_size;

      /* Bound the transient GPU memory: if we'd exceed the ceiling, fall back to
       * the CPU path for this upload rather than risk exhausting device memory. */
      simple_mtx_lock(&device->bcn_gpu_mutex);
      if (device->bcn_gpu_inflight + total > CAP) {
         simple_mtx_unlock(&device->bcn_gpu_mutex);
         return false;
      }
      device->bcn_gpu_inflight += total;
      simple_mtx_unlock(&device->bcn_gpu_mutex);

      struct wrapper_buffer *srcb = wrapper_bcn_make_buffer(device, src_size,
         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      struct wrapper_buffer *dstb = wrapper_bcn_make_buffer(device, dst_size,
         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      if (!srcb || !dstb) {
         wrapper_bcn_free_buffer(device, srcb);
         wrapper_bcn_free_buffer(device, dstb);
         simple_mtx_lock(&device->bcn_gpu_mutex);
         device->bcn_gpu_inflight -= total;
         simple_mtx_unlock(&device->bcn_gpu_mutex);
         return false;
      }
      srcb->bcn_inflight = src_size;
      dstb->bcn_inflight = dst_size;

      VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &psz,
      };
      VkDescriptorPool pool;
      VkDescriptorSet set;
      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorSetCount = 1, .pSetLayouts = &device->bcn_set_layout,
      };
      if (device->dispatch_table.CreateDescriptorPool(device->dispatch_handle,
            &dpci, NULL, &pool) != VK_SUCCESS) {
         wrapper_bcn_free_buffer(device, srcb);
         wrapper_bcn_free_buffer(device, dstb);
         simple_mtx_lock(&device->bcn_gpu_mutex);
         device->bcn_gpu_inflight -= total;
         simple_mtx_unlock(&device->bcn_gpu_mutex);
         return false;
      }
      dsai.descriptorPool = pool;
      if (device->dispatch_table.AllocateDescriptorSets(device->dispatch_handle,
            &dsai, &set) != VK_SUCCESS) {
         device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, pool, NULL);
         wrapper_bcn_free_buffer(device, srcb);
         wrapper_bcn_free_buffer(device, dstb);
         simple_mtx_lock(&device->bcn_gpu_mutex);
         device->bcn_gpu_inflight -= total;
         simple_mtx_unlock(&device->bcn_gpu_mutex);
         return false;
      }
      VkDescriptorBufferInfo bi0 = { srcb->dispatch_handle, 0, src_size };
      VkDescriptorBufferInfo bi1 = { dstb->dispatch_handle, 0, dst_size };
      VkWriteDescriptorSet writes[2] = {
         { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
           .dstBinding = 0, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi0 },
         { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
           .dstBinding = 1, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi1 },
      };
      device->dispatch_table.UpdateDescriptorSets(device->dispatch_handle, 2, writes, 0, NULL);

      VkBufferCopy bcopy = { .srcOffset = offset, .dstOffset = 0, .size = src_size };
      device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle,
         wb->dispatch_handle, srcb->dispatch_handle, 1, &bcopy);

      VkMemoryBarrier mb1 = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      };
      device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         0, 1, &mb1, 0, NULL, 0, NULL);

      device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
         VK_PIPELINE_BIND_POINT_COMPUTE, device->bcn_pipeline);
      device->dispatch_table.CmdBindDescriptorSets(wcb->dispatch_handle,
         VK_PIPELINE_BIND_POINT_COMPUTE, device->bcn_pipe_layout, 0, 1, &set, 0, NULL);
      struct wrapper_bcn_pc pc = {
         .block_x = block_x, .block_x_src = block_x_src, .block_y = block_y,
         .src_word_off = 0, .format = fmt_id, .has_alpha = has_alpha,
         .astc8 = astc8, .bc_bx = bc_bx, .bc_by = bc_by,
      };
      device->dispatch_table.CmdPushConstants(wcb->dispatch_handle,
         device->bcn_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      device->dispatch_table.CmdDispatch(wcb->dispatch_handle,
         (block_x + 7) / 8, (block_y + 7) / 8, 1);

      VkMemoryBarrier mb2 = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      };
      device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
         0, 1, &mb2, 0, NULL, 0, NULL);

      region.bufferOffset = 0;
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;
      device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle,
         dstb->dispatch_handle, dstImage, dstLayout, 1, &region);

      srcb->wcb = wcb;
      dstb->wcb = wcb;
      dstb->desc_pool = pool;
      if (wcb->fence) {
         list_add(&srcb->link, &wcb->fence->staging_buffers_list);
         list_add(&dstb->link, &wcb->fence->staging_buffers_list);
      }
   }
   return true;
}

static void
wrapper_bcn_do_copy(struct wrapper_command_buffer *wcb,
                    struct wrapper_device *device,
                    struct wrapper_buffer *wb,
                    VkImage dstImage,
                    VkImageLayout dstLayout,
                    VkFormat format,
                    uint32_t regionCount,
                    const VkBufferImageCopy *pRegions)
{
   int fmt_id = wrapper_bcn_gpu_format_id(format);
   if (fmt_id >= 0 && wrapper_bcn_gpu_ready(device)) {
      if (wrapper_bcn_gpu_copy(wcb, device, wb, dstImage, dstLayout, format,
            fmt_id, regionCount, pRegions))
         return;
      /* else fall through to CPU transcode */
   }
   VkResult res;

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
      VkDeviceSize upload_size = bcn_upload_size(format, w, h);

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
         simple_mtx_unlock(&device->resource_mutex);
         return;
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
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }

      res = device->dispatch_table.BindBufferMemory(device->dispatch_handle,
         staging_wb->dispatch_handle, staging_wb->memory, 0);

      if (res != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to bind staging buffer memory, res %d", res);
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }

      res = device->dispatch_table.MapMemory(device->dispatch_handle,
         staging_wb->memory, 0, upload_size, 0, &staging_wb->mapped_address);

      if (res != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to map staging buffer memory, res %d", res);
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }

      int src_w = copy_region.bufferRowLength ? (int)copy_region.bufferRowLength : w;
      decompress_bcn_format(wb->mapped_address, staging_wb->mapped_address, w, h, src_w, format, offset);

      /* Texture-path diagnostic (part of WRAPPER_DIAG=1): for each BCn CPU upload,
       * record the real parameters (source stride, target format, image tiling/
       * extent/usage) and dump the leading source BC7 bytes + our transcode
       * output, so the transcode can be verified byte-for-byte offline against
       * the reference decoder and any stride/format/tiling mismatch is visible. */
      {
         static int tex_diag = -1;
         if (tex_diag == -1)
            tex_diag = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
         if (tex_diag) {
            static int tex_n = 0;
            struct wrapper_image *twi = get_wrapper_image_from_handle(device, dstImage);
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
            if (tex_n < 4) {
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
         }
      }

      copy_region.bufferOffset = 0;
      copy_region.bufferRowLength = 0;
      copy_region.bufferImageHeight = 0;

      device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle,
         staging_wb->dispatch_handle, dstImage, dstLayout, 1, &copy_region);

      staging_wb->wcb = wcb;
      staging_wb->device = device;

      if (wcb->fence)
         list_add(&staging_wb->link, &wcb->fence->staging_buffers_list);
   }

   if (wb->is_mapped) {
      device->dispatch_table.UnmapMemory(device->dispatch_handle,
         wb->memory);

      wb->is_mapped = 0;
   }

   simple_mtx_unlock(&device->resource_mutex);
}

/* Append a line to the per-game diag file and mirror to logcat, when
 * WRAPPER_DIAG is set. Used for the copy-level texture log below. */
static void
wrapper_diag_append(const char *fmt, ...)
{
   static int on = -1;
   if (on == -1)
      on = getenv("WRAPPER_DIAG") ? atoi(getenv("WRAPPER_DIAG")) : 0;
   if (!on)
      return;
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

   wrapper_bcn_do_copy(wcb, device, wb, dstImage, dstLayout, format,
      regionCount, pRegions);
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

   wrapper_bcn_do_copy(wcb, device, wb, pInfo->dstImage, pInfo->dstImageLayout,
      format, pInfo->regionCount, regions);
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

   device->dispatch_table.CmdBlitImage(
      wcb->dispatch_handle,
      srcImage, srcImageLayout,
      dstImage, dstImageLayout,
      regionCount, regions, filter);
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

   if (device->dispatch_table.CmdBlitImage2) {
      device->dispatch_table.CmdBlitImage2(wcb->dispatch_handle, pBlitImageInfo);
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
   _mesa_hash_table_u64_destroy(device->image_view_table);
   _mesa_hash_table_u64_destroy(device->imageless_fb_table);

   list_for_each_entry_safe(struct vk_queue, queue, &device->vk.queues, link) {
      vk_queue_finish(queue);
      vk_free2(&device->vk.alloc, pAllocator, queue);
   }
   if (device->dispatch_handle != VK_NULL_HANDLE) {
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
