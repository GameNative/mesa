#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"
#include "vk_common_entrypoints.h"
#include "util/os_file.h"
#include "vk_util.h"

#include <android/hardware_buffer.h>
#include <vndk/hardware_buffer.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>
#include <fcntl.h>
#include <errno.h>

static int
safe_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

   return ret;
}

static int
dma_heap_alloc(int heap_fd, size_t size) {
   struct dma_heap_allocation_data alloc_data = {
      .len = size,
      .fd_flags = O_RDWR | O_CLOEXEC,
   };
   if (safe_ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0)
      return -1;

   return alloc_data.fd;
}

// https://cs.android.com/android/kernel/superproject/+/common-android-4.9:common/drivers/staging/android/uapi/ion.h
struct ion_allocation_data_1 {
	size_t len;
	size_t align;
	unsigned int heap_id_mask;
	unsigned int flags;
	__u32 handle;
};

struct ion_fd_data_1 {
	__u32 handle;
	int fd;
};

struct ion_handle_data_1 {
	__u32 handle;
};

#define ION_IOC_MAGIC       'I'
#define ION_IOC_ALLOC_1       _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data_1)
#define ION_IOC_FREE_1        _IOWR(ION_IOC_MAGIC, 1, struct ion_handle_data_1)
#define ION_IOC_MAP_1         _IOWR(ION_IOC_MAGIC, 2, struct ion_fd_data_1)
#define ION_IOC_SHARE_1		   _IOWR(ION_IOC_MAGIC, 4, struct ion_fd_data_1)

// https://cs.android.com/android/kernel/superproject/+/common-android-4.14:common/drivers/staging/android/uapi/ion.h
struct ion_allocation_data_2 {
   __u64 len;
   __u32 heap_id_mask;
   __u32 flags;
   __u32 fd;
   __u32 unused;
};

struct ion_heap_query_2 {
	__u32 cnt; /* Total number of heaps to be copied */
	__u32 reserved0; /* align to 64bits */
	__u64 heaps; /* buffer to be populated */
	__u32 reserved1;
	__u32 reserved2;
};

#define ION_IOC_ALLOC_2       _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data_2)
#define ION_IOC_HEAP_QUERY_2     _IOWR(ION_IOC_MAGIC, 8, struct ion_heap_query_2)

static int
ion_heap_alloc_2(int heap_fd, size_t size) {
   struct ion_allocation_data_2 alloc_data = {
      .len = size,
      /* ION_HEAP_SYSTEM | ION_SYSTEM_HEAP_ID (Qcom) */
      .heap_id_mask = (1U << 0) | (1U << 25),
      .flags = 0,
   };

   if (safe_ioctl(heap_fd, ION_IOC_ALLOC_2, &alloc_data) < 0) {
      alloc_data.heap_id_mask = 1U;
      if (safe_ioctl(heap_fd, ION_IOC_ALLOC_2, &alloc_data) < 0) {
         return -1;
      }
   }

   return alloc_data.fd;
}

static int
ion_heap_alloc(int heap_fd, size_t size) {
   static int ion_iface = 0;
   if (!ion_iface) {
      // See https://github.com/mirror/mesa/blob/e24dc5bd1e7fe6101bdc866fb16a15a8fcae1aae/src/freedreno/vulkan/tu_knl_kgsl.cc#L1789
      struct ion_handle_data_1 probe = { .handle = 0 };
      if (safe_ioctl(heap_fd, ION_IOC_FREE_1, &probe) >= 0 || errno != ENOTTY) {
         ion_iface = 1;
      } else {
         ion_iface = 2;
      }
      WRAPPER_LOG("info", "Picking ion interface: %d", ion_iface);
   }

   if (ion_iface == 2) {
      return ion_heap_alloc_2(heap_fd, size);
   }

   // see https://github.com/mirror/mesa/blob/e24dc5bd1e7fe6101bdc866fb16a15a8fcae1aae/src/freedreno/vulkan/tu_knl_kgsl.cc#L122
   // bo_init_new_ion_legacy
   struct ion_allocation_data_1 alloc_data = {
      .len = size,
      .align = 0,
      .heap_id_mask = (1U << 0) | (1U << 25) /* QCom specific */,
      .flags = 0,
   };

   if (safe_ioctl(heap_fd, ION_IOC_ALLOC_1, &alloc_data) < 0) {
      alloc_data.align = 4096;
      alloc_data.heap_id_mask = 1U;
      if (safe_ioctl(heap_fd, ION_IOC_ALLOC_1, &alloc_data) < 0) {
         return -1;
      }
   }

   struct ion_fd_data_1 fd_data = {
      .handle = alloc_data.handle,
      .fd = -1,
   };
   if (safe_ioctl(heap_fd, ION_IOC_SHARE_1, &fd_data) < 0) {
      int saved_errno = errno;
      struct ion_handle_data_1 free_data = { .handle = alloc_data.handle };
      safe_ioctl(heap_fd, ION_IOC_FREE_1, &free_data);
      WRAPPER_LOG("error", "Failed to share handle, errno=%d", saved_errno);
      errno = saved_errno;
      return -1;
   }

   struct ion_handle_data_1 free_data = { .handle = alloc_data.handle };
   safe_ioctl(heap_fd, ION_IOC_FREE_1, &free_data);

   return fd_data.fd;
}

static int
wrapper_dmabuf_alloc(struct wrapper_device *device, size_t size)
{
   int fd;

   fd = dma_heap_alloc(device->physical->dma_heap_fd, size);

   if (fd < 0)
      fd = ion_heap_alloc(device->physical->dma_heap_fd, size);

   return fd;
}

uint32_t
wrapper_select_device_memory_type(struct wrapper_device *device,
                                  VkMemoryPropertyFlags flags) {
   VkPhysicalDeviceMemoryProperties *props =
      &device->physical->memory_properties;
   int idx;

   for (idx = 0; idx < props->memoryTypeCount; idx ++) {
      if (props->memoryTypes[idx].propertyFlags & flags) {
         break;
      }
   }
   return idx < props->memoryTypeCount ? idx : UINT32_MAX;
}

static uint32_t
wrapper_select_allowed_device_memory_type(struct wrapper_device *device,
                                          uint32_t allowed_type_bits,
                                          VkMemoryPropertyFlags flags) {
   VkPhysicalDeviceMemoryProperties *props =
      &device->physical->memory_properties;
   int idx;

   for (idx = 0; idx < props->memoryTypeCount; idx++) {
      if (!(allowed_type_bits & (1U << idx))) {
         continue;
      }

      if (props->memoryTypes[idx].propertyFlags & flags) {
         return idx;
      }
   }
   return UINT32_MAX;
}

static inline void
unlink_memory_alloc_info_pnext(VkMemoryAllocateInfo *alloc_info, VkStructureType sType)
{
   const VkBaseInStructure *head = (const VkBaseInStructure *)alloc_info->pNext;
   if (head && head->sType == sType) {
      alloc_info->pNext = head->pNext;
      return;
   }

   VkBaseOutStructure *prev = NULL;
   vk_foreach_struct(s, (void *)alloc_info->pNext) {
      if (s->sType == sType) {
         if (prev) {
            prev->pNext = s->pNext; // modifies application owned memory, but it should be okay
         }
         return;
      }
      prev = s;
   }
}

static VkResult check_dedicated_allocate_info_for(struct wrapper_device *device,
                                              const VkMemoryDedicatedAllocateInfo *memory_dedicated_info,
                                              VkExternalMemoryHandleTypeFlags handle_types) {
   if (!memory_dedicated_info) {
      return VK_SUCCESS;
   }

   if (memory_dedicated_info->image != VK_NULL_HANDLE) {
      struct wrapper_image *img = get_wrapper_image_from_handle_locked(
         device, memory_dedicated_info->image);
      if (img && img->handle_types && !(img->handle_types & handle_types)) {
         // Shouldn't happen anymore
         WRAPPER_LOG(error, "Dedicated image handle type mismatch (0x%x vs required 0x%x)",
                     img ? img->handle_types : 0, handle_types);
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      }
   }

   if (memory_dedicated_info->buffer != VK_NULL_HANDLE) {
      struct wrapper_buffer *buf = get_wrapper_buffer_from_handle_locked(
         device, memory_dedicated_info->buffer);
      if (buf && buf->handle_types && !(buf->handle_types & handle_types)) {
         // Shouldn't happen anymore
         WRAPPER_LOG(error, "Dedicated buffer handle type mismatch (0x%x vs required 0x%x)",
                     buf ? buf->handle_types : 0, handle_types);
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      }
   }

   return VK_SUCCESS;
}

static VkResult
wrapper_allocate_memory_dmaheap(struct wrapper_device *device,
                                const VkMemoryAllocateInfo* pAllocateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkDeviceMemory* pMemory,
                                int *out_fd) {
   VkImportMemoryFdInfoKHR import_fd_info;
   VkMemoryAllocateInfo allocate_info;
   VkResult result;

   *out_fd = wrapper_dmabuf_alloc(device, pAllocateInfo->allocationSize);
   if (*out_fd < 0)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;

   VkMemoryFdPropertiesKHR memory_fd_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
      .pNext = NULL,
   };
   result = device->dispatch_table.GetMemoryFdPropertiesKHR(
      device->dispatch_handle, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
         *out_fd, &memory_fd_props);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to get memory fd properties, res %d", result);
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }
   
   int memory_type_index = wrapper_select_allowed_device_memory_type(device,
      memory_fd_props.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   if (memory_type_index == UINT32_MAX) {
      WRAPPER_LOG(error, "No compatible memory type found for fd %d", *out_fd);
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   const VkMemoryDedicatedAllocateInfo *memory_dedicated_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   result = check_dedicated_allocate_info_for(
      device, memory_dedicated_info, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
   if (result != VK_SUCCESS) {
      return result;
   }

   import_fd_info = (VkImportMemoryFdInfoKHR) {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = pAllocateInfo->pNext,
      .fd = os_dupfd_cloexec(*out_fd),
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   allocate_info = *pAllocateInfo;
   allocate_info.pNext = &import_fd_info;
   allocate_info.memoryTypeIndex = memory_type_index;

   result = device->dispatch_table.AllocateMemory(
      device->dispatch_handle, &allocate_info,
         pAllocator, pMemory);

   if (result != VK_SUCCESS && import_fd_info.fd != -1) {
      WRAPPER_LOG(error, "Failed to import dmaheap memory, res %d", result);
      close(import_fd_info.fd);
   }

   return result;
}

static VkResult
wrapper_allocate_memory_opaque_fd(struct wrapper_device *device,
								  const VkMemoryAllocateInfo *pAllocateInfo,
								  const VkAllocationCallbacks *pAllocator,
								  VkDeviceMemory *pMemory,
								  int *out_fd)
{
   VkResult result;
   VkMemoryAllocateInfo allocate_info;

   const VkMemoryDedicatedAllocateInfo *memory_dedicated_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   result = check_dedicated_allocate_info_for(
      device, memory_dedicated_info, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);
   if (result != VK_SUCCESS) {
      return result;
   }

   VkExportMemoryAllocateInfo export_memory_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = NULL,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };

   allocate_info = *pAllocateInfo;
   allocate_info.pNext = &export_memory_info;

   result = device->dispatch_table.AllocateMemory(device->dispatch_handle,
   												  &allocate_info,
   												  pAllocator,
   												  pMemory);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate opaque fd memory, res %d", result);
      return result;
   }

   VkMemoryGetFdInfoKHR get_memory_fd = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .pNext = NULL,
      .memory = *pMemory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };

   result = device->dispatch_table.GetMemoryFdKHR(device->dispatch_handle,
   							    &get_memory_fd,
   							    out_fd);

   if (result != VK_SUCCESS || *out_fd < 0) {
      WRAPPER_LOG(error, "Failed to get opaque fd");
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   return VK_SUCCESS;
}

static VkResult
wrapper_allocate_memory_ahardware_buffer(struct wrapper_device *device,
                                         const VkMemoryAllocateInfo* pAllocateInfo,
                                         const VkAllocationCallbacks* pAllocator,
                                         VkDeviceMemory* pMemory,
                                         AHardwareBuffer **pAHardwareBuffer) {
   VkExportMemoryAllocateInfo export_memory_info;
   VkMemoryAllocateInfo allocate_info;
   VkResult result;

   const VkMemoryDedicatedAllocateInfo *memory_dedicated_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   result = check_dedicated_allocate_info_for(
      device, memory_dedicated_info, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID);
   if (result != VK_SUCCESS) {
      return result;
   }

   export_memory_info = (VkExportMemoryAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = pAllocateInfo->pNext,
      .handleTypes =
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
   };
   allocate_info = *pAllocateInfo;
   allocate_info.pNext = &export_memory_info;
  
   if (memory_dedicated_info && memory_dedicated_info->image != VK_NULL_HANDLE) {
      WRAPPER_LOG(info, "VkMemoryDedicatedInfo struct with a non NULL image detected, patching allocationSize");
      allocate_info.allocationSize = 0;
   }
   
   result = device->dispatch_table.AllocateMemory(device->dispatch_handle,
                                                  &allocate_info,
                                                  pAllocator,
                                                  pMemory);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate ahb memory, res %d", result);
      return result;
   }
   
   result = device->dispatch_table.GetMemoryAndroidHardwareBufferANDROID(
      device->dispatch_handle,
      &(VkMemoryGetAndroidHardwareBufferInfoANDROID) {
         .sType =
            VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
         .memory = *pMemory,
      },
      pAHardwareBuffer);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to import ahb, res %d", result);
      return result;
   }
   
   if (AHardwareBuffer_getNativeHandle(*pAHardwareBuffer) == NULL) {
      WRAPPER_LOG(error, "Invalid native handle");
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   return VK_SUCCESS;
}

static void
wrapper_device_memory_reset(struct wrapper_device_memory *mem) {
   struct wrapper_device *device = mem->device;
   if (mem->ahardware_buffer) {
      AHardwareBuffer_release(mem->ahardware_buffer);
      mem->ahardware_buffer = NULL;
   }
   if (mem->fd != -1) {
      close(mem->fd);
      mem->fd = -1;
   }
   if (mem->map_address && mem->map_size) {
      munmap(mem->map_address, mem->map_size);
      mem->map_address = NULL;
   }
   if (mem->dispatch_handle != VK_NULL_HANDLE) {
      device->dispatch_table.FreeMemory(device->dispatch_handle,
         mem->dispatch_handle, mem->alloc);
      mem->dispatch_handle = VK_NULL_HANDLE;
   }
}

VkResult
wrapper_device_memory_create(struct wrapper_device *device,
                             const VkAllocationCallbacks *alloc,
                             struct wrapper_device_memory **out_mem)
{
   *out_mem = vk_zalloc2(&device->vk.alloc, alloc,
                         sizeof(struct wrapper_device_memory),
                         8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (*out_mem == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   (*out_mem)->fd = -1;
   (*out_mem)->device = device;
   (*out_mem)->alloc = alloc ? alloc : &device->vk.alloc;
   list_add(&(*out_mem)->link, &device->device_memory_list);
   return VK_SUCCESS;
}

void
wrapper_device_memory_destroy(struct wrapper_device_memory *mem) {
   wrapper_device_memory_reset(mem);
   list_del(&mem->link);
   vk_free2(&mem->device->vk.alloc, mem->alloc, mem);
}

static struct wrapper_device_memory *
wrapper_device_memory_from_handle(struct wrapper_device *device,
                                  VkDeviceMemory handle) {
   struct wrapper_device_memory *mem = NULL;

   simple_mtx_lock(&device->resource_mutex);

   list_for_each_entry(struct wrapper_device_memory, data,
                       &device->device_memory_list, link) {
      if (data->dispatch_handle == handle) {
         mem = data;
      }
   }

   simple_mtx_unlock(&device->resource_mutex);
   return mem;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateMemory(VkDevice _device,
                       const VkMemoryAllocateInfo* pAllocateInfo,
                       const VkAllocationCallbacks* pAllocator,
                       VkDeviceMemory* pMemory) {
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem;
   VkResult result;

   VkMemoryPropertyFlags property_flags =
      device->physical->memory_properties.memoryTypes[
         pAllocateInfo->memoryTypeIndex].propertyFlags;
   
   if (!(property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      goto fallback;
    
   if (!device->vk.enabled_features.memoryMapPlaced ||
       !device->vk.enabled_extensions.EXT_map_memory_placed)
      goto fallback;
      
   if (vk_find_struct_const(pAllocateInfo, IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID))
      goto fallback;

   if (vk_find_struct_const(pAllocateInfo, IMPORT_MEMORY_FD_INFO_KHR))
      goto fallback;

   if (vk_find_struct_const(pAllocateInfo, EXPORT_MEMORY_ALLOCATE_INFO))
      goto fallback;

   const VkMemoryDedicatedAllocateInfo *dedicated_allocate_info =
         vk_find_struct_const((void*) pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   
   static int bypass_swapchains = -1;
   if (bypass_swapchains == -1) 
      bypass_swapchains = getenv("WRAPPER_BYPASS_SWAPCHAIN_PLACED") ?
                          atoi(getenv("WRAPPER_BYPASS_SWAPCHAIN_PLACED")) : 0; // TODO: turn on by default if safe

   if (bypass_swapchains && dedicated_allocate_info && dedicated_allocate_info->image != VK_NULL_HANDLE) {
      struct wrapper_image *img = get_wrapper_image_from_handle(device, dedicated_allocate_info->image);
      if (img && img->is_wsi_image) {
         WRAPPER_LOG(info, "Bypassing EXT_map_memory_placed emulation for swapchain image");
         goto fallback;
      }
   }
   
   WRAPPER_LOG(info, "Emulating vkAllocateMemory");

   simple_mtx_lock(&device->resource_mutex);

   result = wrapper_device_memory_create(device, pAllocator, &mem);
   if (result != VK_SUCCESS) {
      vk_error(device, result);
      goto out;
   }
   mem->alloc_size = pAllocateInfo->allocationSize;

   VkExternalMemoryHandleTypeFlags valid_handle_types = 0;
   if (dedicated_allocate_info) {
      // Note that buffer/image are mutually exclusive
      if (dedicated_allocate_info->image != VK_NULL_HANDLE) {
         struct wrapper_image *img = get_wrapper_image_from_handle_locked(device, dedicated_allocate_info->image);
         if (img) {
            valid_handle_types |= img->handle_types;
         }
      }
      if (dedicated_allocate_info->buffer != VK_NULL_HANDLE) {
         struct wrapper_buffer *buf = get_wrapper_buffer_from_handle_locked(device, dedicated_allocate_info->buffer);
         if (buf) {
            valid_handle_types |= buf->handle_types;
         }
      }
   }

   VkMemoryAllocateInfo memory_allocate_info = *pAllocateInfo;
   if (dedicated_allocate_info && valid_handle_types == 0) {
      // Driver "bug" on some mobile drivers - providing an empty dedicate memory hint in conjunction with the
      // VkImportMemoryFdInfoKHR / VkExportMemoryAllocateInfo could crash the driver
      unlink_memory_alloc_info_pnext(&memory_allocate_info, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
   }
   
   if (strstr(device->physical->resource_type, "ahb")) {
      WRAPPER_LOG(info, "Using AHardwareBuffer memory backend");
      result = wrapper_allocate_memory_ahardware_buffer(device,
         &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->ahardware_buffer);
   }
   else if (strstr(device->physical->resource_type, "dmabuf")) {
      WRAPPER_LOG(info, "Using DMABUF memory backend");
      result = wrapper_allocate_memory_dmaheap(device,
         &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->fd);
   }
   else if (strstr(device->physical->resource_type, "opaque")) {
      WRAPPER_LOG(info, "Using opaque fd memory backend");
      result = wrapper_allocate_memory_opaque_fd(device,
         &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->fd);
   }
   else {
      WRAPPER_LOG(info, "Using auto memory backend");
      result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
#define VALID_HANDLE(type) (valid_handle_types == 0 || (type & valid_handle_types) != 0)
      if (VALID_HANDLE(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) {
         result = wrapper_allocate_memory_dmaheap(device,
            &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->fd);
      }

      if (result != VK_SUCCESS && VALID_HANDLE(VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID)) {
         wrapper_device_memory_reset(mem);
         result = wrapper_allocate_memory_ahardware_buffer(device,
            &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->ahardware_buffer);
      }

      if (result != VK_SUCCESS && VALID_HANDLE(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)) {
         wrapper_device_memory_reset(mem);
         result = wrapper_allocate_memory_opaque_fd(device,
            &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->fd);
      }
#undef VALID_HANDLE
   }
   
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate memory, res %d", result);
      wrapper_device_memory_destroy(mem);

      if (dedicated_allocate_info && dedicated_allocate_info->image != VK_NULL_HANDLE) {
         /* resource_mutex is already held here and simple_mtx is not
          * recursive, so search the image table directly instead of
          * going through get_wrapper_image_from_handle. */
         struct wrapper_image *img = get_wrapper_image_from_handle_locked(
            device, dedicated_allocate_info->image);
         if (img && img->is_wsi_image) {
            // Fixes failure to blit on ion-heap (< GKI 5.10) Mali devices at the cost of
            // not being able to mmap these.
            WRAPPER_LOG(error, "EXT_map_memory_placed emulation failed for swapchain image, bypassing emulation");
            simple_mtx_unlock(&device->resource_mutex);
            goto fallback; // TODO: the VkMemoryAllocateInfo may have been unlinked here
         }
      }

      vk_error(device, result);
   } else {
      *pMemory = mem->dispatch_handle;
   }

out:
   simple_mtx_unlock(&device->resource_mutex);
   return result;

fallback:
   return device->dispatch_table.AllocateMemory(device->dispatch_handle,
      pAllocateInfo, pAllocator, pMemory);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_FreeMemory(VkDevice _device, VkDeviceMemory _memory,
                   const VkAllocationCallbacks* pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem;

   mem = wrapper_device_memory_from_handle(device, _memory);
   if (mem) {
      mem->alloc = pAllocator;
      return wrapper_device_memory_destroy(mem);
   }

   device->dispatch_table.FreeMemory(device->dispatch_handle,
                                     _memory,
                                     pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_MapMemory2KHR(VkDevice _device,
                      const VkMemoryMapInfoKHR* pMemoryMapInfo,
                      void** ppData)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkResult result;
   const VkMemoryMapPlacedInfoEXT *placed_info = NULL;
   struct wrapper_device_memory *mem;
   int fd;

   if (pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT)
      placed_info = vk_find_struct_const(pMemoryMapInfo->pNext,
         MEMORY_MAP_PLACED_INFO_EXT);
   
   mem = wrapper_device_memory_from_handle(device, pMemoryMapInfo->memory);
   if (!placed_info || !mem) {
      return device->dispatch_table.MapMemory(device->dispatch_handle,
         pMemoryMapInfo->memory, pMemoryMapInfo->offset, pMemoryMapInfo->size,
            0, ppData);
   }

   WRAPPER_LOG(info, "Emulating vkMapMemory2KHR");

   simple_mtx_lock(&device->resource_mutex);

   if (mem->map_address) {
      if (placed_info->pPlacedAddress != mem->map_address) {
         WRAPPER_LOG(error, "Placed address/mapped address mismatch");
         result = VK_ERROR_MEMORY_MAP_FAILED;
         goto fail;
      } else {
         goto out;
      }
   }
   assert(mem->fd >= 0 || mem->ahardware_buffer != NULL);

   if (mem->ahardware_buffer) {
      const native_handle_t *handle;
      int idx;

      handle = AHardwareBuffer_getNativeHandle(mem->ahardware_buffer);
      /* The AHB native handle may carry several fds (e.g. metadata pipes
       * alongside the actual memory fd); pick the first one that is
       * seekable and large enough to back the allocation. */
      for (idx = 0; idx < handle->numFds; idx++) {
         off_t size = lseek(handle->data[idx], 0, SEEK_END);
         if (size < 0) {
            WRAPPER_LOG(error, "lseek failed on AHB fd (idx=%d, fd=%d): errno %d, trying next fd",
                        idx, handle->data[idx], errno);
            continue;
         }
         if ((size_t)size >= mem->alloc_size)
            break;
      }
      if (idx >= handle->numFds) {
         WRAPPER_LOG(error, "No usable AHB fd with size >= alloc_size %zu", mem->alloc_size);
         result = VK_ERROR_MEMORY_MAP_FAILED;
         goto fail;
      }
      fd = handle->data[idx];
   }
   else {
      fd = mem->fd;
   }

   if (pMemoryMapInfo->size == VK_WHOLE_SIZE) {
      if (mem->alloc_size > 0) {
         mem->map_size = mem->alloc_size;
      } else {
         off_t res = lseek(fd, 0, SEEK_END);
         if (res < 0) {
            WRAPPER_LOG(error, "Failed lseek for file descriptor %d: errno %d", fd, errno);
            result = VK_ERROR_MEMORY_MAP_FAILED;
            goto fail;
         }
         mem->map_size = res;
      }
   }
   else
      mem->map_size = pMemoryMapInfo->size;

   WRAPPER_LOG(info, "Mapping memory %p, address %p size %zu\n", pMemoryMapInfo->memory, placed_info->pPlacedAddress, mem->map_size);

   mem->map_address = mmap(placed_info->pPlacedAddress,
      mem->map_size, PROT_READ | PROT_WRITE,
         MAP_SHARED | MAP_FIXED, fd, 0);

   if (mem->map_address == MAP_FAILED) {
      WRAPPER_LOG(error, "mmap failed: error %d", errno);
      mem->map_address = NULL;
      mem->map_size = 0;
      result = VK_ERROR_MEMORY_MAP_FAILED;
      goto fail;
   }

   out:
      simple_mtx_unlock(&device->resource_mutex);
      *ppData = (char *)mem->map_address + pMemoryMapInfo->offset;
      return VK_SUCCESS;
   fail:
      simple_mtx_unlock(&device->resource_mutex);
      return result;
}

VKAPI_ATTR void VKAPI_CALL
wrapper_UnmapMemory(VkDevice _device, VkDeviceMemory _memory) {
   vk_common_UnmapMemory(_device, _memory);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_UnmapMemory2KHR(VkDevice _device,
                        const VkMemoryUnmapInfoKHR* pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem;

   mem = wrapper_device_memory_from_handle(device, pMemoryUnmapInfo->memory);
   if (!mem) {
      device->dispatch_table.UnmapMemory(device->dispatch_handle,
         pMemoryUnmapInfo->memory);
      return VK_SUCCESS;
   }

   WRAPPER_LOG(info, "Emulating vkUnmapMemory2KHR");

   if (pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT) {
      mem->map_address = mmap(mem->map_address, mem->map_size,
         PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (mem->map_address == MAP_FAILED) {
         WRAPPER_LOG(error, "Failed to replace mapping with reserved memory");
         return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);
      }
   } else {
      munmap(mem->map_address, mem->map_size);
   }

   mem->map_size = 0;
   mem->map_address = NULL;
   return VK_SUCCESS;
}
