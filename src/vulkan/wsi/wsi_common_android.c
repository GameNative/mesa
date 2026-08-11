#include "wsi_common.h"
#include "wsi_common_private.h"
#include "vk_log.h"
#include "../wrapper/wrapper_log.h"
#include "../wrapper/wrapper_private.h"

#include <android/hardware_buffer.h>

#define AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM 1
#define AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM 5

static enum wsi_swapchain_blit_type
wsi_get_ahardware_buffer_blit_type(const struct wsi_device *wsi,
                            VkDevice device)
{
   AHardwareBuffer *ahardware_buffer;
   VkResult result;
   uint32_t probe_format = wsi->emulate_bgra8
         ? AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM
         : AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM;
   
   if (AHardwareBuffer_allocate(&(AHardwareBuffer_Desc){
      .width = 500,
      .height = 500,
      .layers = 1,
      .format = probe_format,
      .usage = AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER |
               AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
               AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
               AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN },
                                &ahardware_buffer) != 0) {
      WRAPPER_LOG(error, "Failed to allocate ahardware buffer, blitting");
      return WSI_SWAPCHAIN_IMAGE_BLIT;
   }

   VkAndroidHardwareBufferFormatPropertiesANDROID ahardware_buffer_format_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
      .pNext = NULL,
   };
   VkAndroidHardwareBufferPropertiesANDROID ahardware_buffer_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
      .pNext = &ahardware_buffer_format_props,
   };
   result = wsi->GetAndroidHardwareBufferPropertiesANDROID(
      device, ahardware_buffer, &ahardware_buffer_props);

   AHardwareBuffer_release(ahardware_buffer);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to get ahardware buffer properties, blitting");
      return WSI_SWAPCHAIN_IMAGE_BLIT;
   }

   VkPhysicalDeviceExternalImageFormatInfo external_format_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
      .pNext = NULL,
      .handleType =
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
   };
   VkPhysicalDeviceImageFormatInfo2 format_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext = &external_format_info,
      .format = ahardware_buffer_format_props.format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
      .flags = 0u,
   };
   VkExternalImageFormatProperties external_format_props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
      .pNext = NULL,
   };
   VkImageFormatProperties2 format_props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &external_format_props,
   };
   result = wsi->GetPhysicalDeviceImageFormatProperties2(
      wsi->pdevice, &format_info, &format_props);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "External Image format not supported, blitting");
      return WSI_SWAPCHAIN_IMAGE_BLIT;
   }

   if (!(external_format_props.externalMemoryProperties.externalMemoryFeatures
         & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
      WRAPPER_LOG(error, "External image format isn't importable, blitting");
      return WSI_SWAPCHAIN_IMAGE_BLIT;
   }

   WRAPPER_LOG(info, "wsi_get_ahardware_buffer_blit_type: WSI_SWAPCHAIN_NO_BLIT");
   return WSI_SWAPCHAIN_NO_BLIT;
}


enum wsi_swapchain_blit_type
wsi_get_android_blit_type(const struct wsi_device *wsi,
                      const struct wsi_base_image_params *params,
                                   VkDevice device)
{
   int wrapper_blit = getenv("WRAPPER_BLIT") && atoi(getenv("WRAPPER_BLIT"));
   if (wsi->needs_blit || wrapper_blit)
      return WSI_SWAPCHAIN_IMAGE_BLIT;

   return wsi_get_ahardware_buffer_blit_type(wsi, device);
}

static VkResult
wsi_create_ahardware_buffer_image_mem(const struct wsi_swapchain *chain,
                                      const struct wsi_image_info *info,
                                      struct wsi_image *image)
{
   const struct wsi_device *wsi = chain->wsi;
   VkImage old_image = image->image;
   VkResult result;

   WRAPPER_LOG(info, "Allocating memory for %dx%d image with format %d, usage %d, flags %d",
      info->create.extent.width, info->create.extent.height, 
      info->create.format, info->create.usage, 
      info->create.flags);

   VkAndroidHardwareBufferFormatPropertiesANDROID ahardware_buffer_format_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
      .pNext = NULL,
   };
   VkAndroidHardwareBufferPropertiesANDROID ahardware_buffer_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
      .pNext = &ahardware_buffer_format_props,
   };
   result = wsi->GetAndroidHardwareBufferPropertiesANDROID(
      chain->device, image->ahardware_buffer, &ahardware_buffer_props);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to get ahardware buffer properties, res %d", result);
      return result;
   }

   VkImageCreateInfo new_image_create_info = info->create;
   if (ahardware_buffer_format_props.externalFormat && !wsi->emulate_bgra8)
      new_image_create_info.flags &=
         ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   new_image_create_info.format = ahardware_buffer_format_props.format;

   VkEmulatedB8G8R8A8CreateInfoExt emulated_bgra8_ext = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EMULATED_B8G8R8A8_CREATE_INFO_EXT,
      .pNext = new_image_create_info.pNext,
   };
   if (wsi->emulate_bgra8) {
      new_image_create_info.pNext = &emulated_bgra8_ext;
   }

   result = wsi->CreateImage(chain->device,
                             &new_image_create_info,
                             &chain->alloc, &image->image);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create image, res %d", result);
      return result;
   }

   wsi->DestroyImage(chain->device, old_image, &chain->alloc);

   const VkMemoryDedicatedAllocateInfo memory_dedicated_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image->image,
      .buffer = VK_NULL_HANDLE,
   };
   VkImportAndroidHardwareBufferInfoANDROID import_ahardware_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
      .pNext = &memory_dedicated_info,
      .buffer = image->ahardware_buffer,
   };
   VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import_ahardware_buffer_info,
      .allocationSize = ahardware_buffer_props.allocationSize,
      .memoryTypeIndex =
         wsi_select_device_memory_type(
         wsi, ahardware_buffer_props.memoryTypeBits),
   };

   result = wsi->AllocateMemory(chain->device, &memory_info,
                                &chain->alloc, &image->memory);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate image memory, res %d", result);
   }

   image->num_planes = 1;
   image->drm_modifier = 1255;

   return VK_SUCCESS;
}

/* Release the image to an external consumer.
 *
 * The direct AHardwareBuffer path exports an image that another Vulkan device
 * may read. Vulkan models that handoff as a queue-family ownership transfer.
 * The producing device must perform the release so driver-private image state
 * is made available to the importer.
 */
static VkResult
wsi_create_ahb_release_cmd_buffers(const struct wsi_swapchain *chain,
                                   struct wsi_image *image)
{
   const struct wsi_device *wsi = chain->wsi;
   VkResult result;

   const uint32_t count = wsi->queue_family_count;
   image->ahb_release_cmd_buffers =
      vk_zalloc(&chain->alloc, sizeof(VkCommandBuffer) * count, 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image->ahb_release_cmd_buffers)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   for (uint32_t i = 0; i < count; i++) {
      if (!chain->cmd_pools[i])
         continue;

      const VkCommandBufferAllocateInfo cmd_buffer_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = chain->cmd_pools[i],
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      result = wsi->AllocateCommandBuffers(chain->device, &cmd_buffer_info,
                                           &image->ahb_release_cmd_buffers[i]);
      if (result != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to allocate ahb release cmd buffer, res %d",
                     result);
         return result;
      }

      const VkCommandBufferBeginInfo begin_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      };
      result = wsi->BeginCommandBuffer(image->ahb_release_cmd_buffers[i],
                                       &begin_info);
      if (result != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to begin ahb release cmd buffer, res %d",
                     result);
         return result;
      }

      const VkImageMemoryBarrier release = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         /* The WSI does not know how the application produced the image.
          * Cover every write rather than assuming color-attachment or blit
          * presentation. */
         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
         /* Ignored for a release operation. */
         .dstAccessMask = 0,
         .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
         /* The importer has to begin its acquire from the layout this release
          * ends in, and it cannot name PRESENT_SRC_KHR for an image that does
          * not belong to its own swapchain. GENERAL is valid on both sides and,
          * unlike UNDEFINED, keeps the contents -- which is the whole point of
          * handing the image over. */
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = i,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
         .image = image->image,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      };
      wsi->CmdPipelineBarrier(image->ahb_release_cmd_buffers[i],
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0, 0, NULL, 0, NULL, 1, &release);

      result = wsi->EndCommandBuffer(image->ahb_release_cmd_buffers[i]);
      if (result != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to record ahb release barrier, res %d",
                     result);
         return result;
      }
   }

   WRAPPER_LOG(info, "Recorded AHB ownership release barriers");
   return VK_SUCCESS;
}

static VkResult
wsi_finish_create_ahardware_buffer_image(const struct wsi_swapchain *chain,
                                         const struct wsi_image_info *info,
                                         struct wsi_image *image)
{
   return wsi_create_ahb_release_cmd_buffers(chain, image);
}

static VkResult
wsi_create_ahardware_buffer_blit_context(const struct wsi_swapchain *chain,
                                         const struct wsi_image_info *info,
                                         struct wsi_image *image)
{
   assert(chain->blit.type == WSI_SWAPCHAIN_IMAGE_BLIT);
   const struct wsi_device *wsi = chain->wsi;
   VkResult result;

   WRAPPER_LOG(info, "Creating blit context for %dx%d image with format %d, usage %d, flags %d",
      info->create.extent.width, info->create.extent.height,
      info->create.format, info->create.usage,
      info->create.flags);
   
   const VkExternalMemoryHandleTypeFlags handle_types =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

   VkAndroidHardwareBufferFormatPropertiesANDROID ahardware_buffer_format_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
      .pNext = NULL,
   };
   VkAndroidHardwareBufferPropertiesANDROID ahardware_buffer_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
      .pNext = &ahardware_buffer_format_props,
   };
   result = wsi->GetAndroidHardwareBufferPropertiesANDROID(
      chain->device, image->ahardware_buffer, &ahardware_buffer_props);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to get ahardware buffer properties, res %d", result);
      return result;
   }

   const VkExternalMemoryImageCreateInfo external_memory_info = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = handle_types,
   };
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &external_memory_info,
      .flags = 0u,
      .extent = info->create.extent,
      .format = ahardware_buffer_format_props.format,
      .imageType = VK_IMAGE_TYPE_2D,
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount =
         info->create.queueFamilyIndexCount,
      .pQueueFamilyIndices =
         info->create.pQueueFamilyIndices,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   result = wsi->CreateImage(chain->device, &image_info,
                             &chain->alloc, &image->blit.image);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to create blit image, res %d", result);
      return result;
   }

   VkMemoryDedicatedAllocateInfo blit_mem_dedicated_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .pNext = NULL,
      .image = image->blit.image,
      .buffer = VK_NULL_HANDLE,
   };
   VkImportAndroidHardwareBufferInfoANDROID import_ahardware_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
      .pNext = &blit_mem_dedicated_info,
      .buffer = image->ahardware_buffer,
   };
   VkMemoryAllocateInfo blit_mem_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import_ahardware_buffer_info,
      .allocationSize = ahardware_buffer_props.allocationSize,
      .memoryTypeIndex =
         wsi_select_device_memory_type(
         wsi, ahardware_buffer_props.memoryTypeBits),
   };
   
   result = wsi->AllocateMemory(chain->device, &blit_mem_info,
                                &chain->alloc, &image->blit.memory);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate blit memory, res %d", result);
      return result;
   }

   result = wsi->BindImageMemory(chain->device, image->blit.image,
                                 image->blit.memory, 0);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to bind image memory, res %d", result);
      return result;
   }

   VkMemoryRequirements reqs;
   wsi->GetImageMemoryRequirements(chain->device, image->image, &reqs);

   const VkMemoryDedicatedAllocateInfo memory_dedicated_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image->image,
   };
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &memory_dedicated_info,
      .allocationSize = reqs.size,
      .memoryTypeIndex =
         wsi_select_device_memory_type(wsi, reqs.memoryTypeBits),
   };
   
   result = wsi->AllocateMemory(chain->device, &memory_info,
                                &chain->alloc, &image->memory);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate image memory, res %d", result);
      return result;
   }
   
   image->num_planes = 1;
   image->drm_modifier = 1255;

   return VK_SUCCESS;
}

inline static uint32_t
to_ahardware_buffer_format(VkFormat format) {
   switch (format) {
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_R8G8B8A8_UNORM:
      return AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
      return AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM;
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
      return AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM;
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM;
   default:
      unreachable("unsupported format");
   }
}

static VkResult
wsi_configure_ahardware_buffer_image(const struct wsi_swapchain *chain,
                                     const VkSwapchainCreateInfoKHR *pCreateInfo,
                                     const bool blit,
                                     struct wsi_image_info *info)
{
   VkResult result;

   VkExternalMemoryHandleTypeFlags handle_type =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
   
   result = wsi_configure_image(chain, pCreateInfo,
                                blit ? 0 : handle_type, info);
   if (result != VK_SUCCESS)
      return result;
   
   info->ahardware_buffer_desc = vk_zalloc(&chain->alloc,
      sizeof(AHardwareBuffer_Desc), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!info->ahardware_buffer_desc) {
      wsi_destroy_image_info(chain, info);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   // WARNING: The secondary AHB MUST be declared as AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM
   // to allow certain mobile drivers to import it, but X11 treats its memory backing 
   // as physical [B, G, R, A] (in reverse order).
   //
   // We CANNOT use vkCmdBlitImage: Vulkan will detect the format mismatch between 
   // B8G8R8A8 (primary) and R8G8B8A8 (secondary) and perform an automatic channel swizzle 
   // [B,G,R,A] -> [R,G,B,A], which breaks X11's expected layout.
   //
   // We MUST use vkCmdCopyImage for a pure bitwise copy, and force the primary image 
   // to B8G8R8A8 upfront so the rendered memory is already in [B, G, R, A] order.
   // 
   // Similarly, we CANNOT advertise an [R,G,B,A] format for the primary image without
   // a design to copy+swizzle the blit from the primary to the secondary.
   *info->ahardware_buffer_desc = (AHardwareBuffer_Desc) {
      .width = pCreateInfo->imageExtent.width,
      .height = pCreateInfo->imageExtent.height,
      .layers = pCreateInfo->imageArrayLayers,
      .format = blit
         ? AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM
         : to_ahardware_buffer_format(info->create.format),
      .usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
               AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER |
               AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
               AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
   };

   return VK_SUCCESS;
}

VkResult
wsi_configure_android_image(
   const struct wsi_swapchain *chain,
   const VkSwapchainCreateInfoKHR *pCreateInfo,
   const struct wsi_base_image_params *params,
   struct wsi_image_info *info)
{
   assert(params->image_type == WSI_IMAGE_TYPE_ANDROID);
   assert(chain->blit.type == WSI_SWAPCHAIN_NO_BLIT ||
          chain->blit.type == WSI_SWAPCHAIN_IMAGE_BLIT);

   VkResult result;

   const bool blit = chain->blit.type == WSI_SWAPCHAIN_IMAGE_BLIT;

   if ((result = wsi_configure_ahardware_buffer_image(chain, pCreateInfo, blit, info)) != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to configure ahardware buffer image, res %d", result);
      return result;
   }

   if (blit) {
      wsi_configure_image_blit_image(chain, info);
      info->create_mem = wsi_create_ahardware_buffer_blit_context;
   } else {
      info->create_mem = wsi_create_ahardware_buffer_image_mem;
      /* Legacy VkImageMemoryBarrier only permits FOREIGN_EXT ownership
       * transfers for exclusive images. Preserve the existing concurrent
       * path instead of recording an invalid barrier for it. */
      if (info->create.sharingMode == VK_SHARING_MODE_EXCLUSIVE)
         info->finish_create = wsi_finish_create_ahardware_buffer_image;
   }

   return VK_SUCCESS;
}
