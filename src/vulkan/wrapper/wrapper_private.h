#ifndef __WRAPPER_PRIVATE_H
#define __WRAPPER_PRIVATE_H

#include "vulkan/runtime/vk_instance.h"
#include "vulkan/runtime/vk_physical_device.h"
#include "vulkan/runtime/vk_device.h"
#include "vulkan/runtime/vk_queue.h"
#include "vulkan/runtime/vk_image.h"
#include "vulkan/runtime/vk_fence.h"
#include "vulkan/runtime/vk_command_buffer.h"
#include "vulkan/runtime/vk_log.h"
#include "vulkan/runtime/vk_buffer.h"
#include "vulkan/util/vk_dispatch_table.h"
#include "vulkan/wsi/wsi_common.h"
#include "util/simple_mtx.h"
#include "util/hash_table.h"

/* Limit advertised for emulated VK_KHR_push_descriptor. 32 is the common
 * hardware value and comfortably covers what vkd3d/DXVK push. */
#define WRAPPER_MAX_PUSH_DESCRIPTORS 32

extern const struct vk_instance_extension_table wrapper_instance_extensions;
extern const struct vk_device_extension_table wrapper_device_extensions;
extern const struct vk_device_extension_table wrapper_filter_extensions;

struct wrapper_instance {
   struct vk_instance vk;

   VkInstance dispatch_handle;
   struct vk_instance_dispatch_table dispatch_table;
};

VK_DEFINE_HANDLE_CASTS(wrapper_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)

struct wrapper_physical_device {
   struct vk_physical_device vk;

   int dma_heap_fd;
   int emulate_bcn;
   uint32_t native_bcn_mask;
   bool is_vkd3d;
   bool emulate_imageless_framebuffer;
   bool emulate_vulkan13;
   char *resource_type;
   VkPhysicalDevice dispatch_handle;
   VkPhysicalDeviceProperties2 properties2;
   VkPhysicalDeviceDriverProperties driver_properties;
   VkPhysicalDeviceMemoryProperties memory_properties;
   struct wsi_device wsi_device;
   struct wrapper_instance *instance;
   struct vk_features base_supported_features;
   struct vk_device_extension_table base_supported_extensions;
   struct vk_physical_device_dispatch_table dispatch_table;
};

VK_DEFINE_HANDLE_CASTS(wrapper_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

struct wrapper_queue {
   struct vk_queue vk;

   struct wrapper_device *device;
   VkQueue dispatch_handle;
};

VK_DEFINE_HANDLE_CASTS(wrapper_queue, vk.base, VkQueue,
                       VK_OBJECT_TYPE_QUEUE)

struct wrapper_device {
   struct vk_device vk;

   VkDevice dispatch_handle;
   simple_mtx_t resource_mutex;
   struct list_head command_buffer_list;
   struct list_head device_memory_list;
   struct list_head buffer_list;
   struct list_head image_list;
   struct list_head fence_list;
   struct hash_table_u64 *buffer_table;
   struct hash_table_u64 *image_table;
   struct hash_table_u64 *image_view_table;
   struct hash_table_u64 *imageless_fb_table;
   struct hash_table_u64 *dynamic_pipeline_table;
   struct hash_table_u64 *fence_table;
   struct wrapper_physical_device *physical;
   struct vk_device_dispatch_table dispatch_table;

   bool emulate_null_descriptor;
   bool device_fault_enabled;
   bool emulate_imageless_framebuffer;

   /* VK_KHR_push_descriptor emulation (for drivers lacking it, e.g. Mali r44).
    * Enabled when the app uses push descriptors and either the base driver
    * lacks the extension or WRAPPER_EMULATE_PUSH_DESCRIPTOR forces it. */
   bool emulate_push_descriptor;
   uint32_t max_push_descriptors;
   simple_mtx_t push_mutex;
   struct hash_table_u64 *push_dsl_table;       /* VkDescriptorSetLayout -> wrapper_push_dsl */
   struct hash_table_u64 *push_pl_table;        /* VkPipelineLayout -> wrapper_push_pl */
   struct hash_table_u64 *push_template_table;  /* VkDescriptorUpdateTemplate -> wrapper_push_template */

   VkBuffer null_buffer;
   VkDeviceMemory null_buffer_memory;
   VkImage null_image;
   VkDeviceMemory null_image_memory;
   VkImageView null_image_view;
   VkSampler null_sampler;

   /* BC decode/encode and ASTC encode compute shaders, lazily initialized. */
   simple_mtx_t bcn_gpu_mutex;
   bool bcn_bda;                         /* bufferDeviceAddress enabled for them */
   int bcn_tool_state;                   /* 0 uninit, 1 ready, -1 disabled */
   VkPipelineLayout bcn_tool_layout;
   VkShaderModule bcn_decode_module;
   VkShaderModule bcn_encode_module;
   VkShaderModule bcn_astc_module;
   VkShaderModule bcn_astc_hdr_module;
   VkPipeline bcn_decode_pipeline;
   VkPipeline bcn_encode_pipeline;
   VkPipeline bcn_astc_pipeline[3];        /* 4x4, 6x6, 8x8 */
   VkPipeline bcn_astc_hdr_pipeline;       /* 4x4 HDR, for BC6H */
   struct bcn_hq *bcn_hq;                  /* background astcenc stage */
   uint32_t bcn_hq_active;                 /* its jobs not yet copied or dropped */

   /* Private queue and synchronous command buffer for host query reset. */
   simple_mtx_t query_reset_mutex;
   VkCommandPool query_reset_pool;
   VkCommandBuffer query_reset_cmd;
   VkFence query_reset_fence;
   VkQueue query_reset_queue;
   uint32_t query_reset_queue_family;
   uint32_t query_reset_queue_index;
   bool query_reset_shared_queue;        /* no spare queue: shares the app's */
};

VK_DEFINE_HANDLE_CASTS(wrapper_device, vk.base, VkDevice,
                       VK_OBJECT_TYPE_DEVICE)

struct wrapper_buffer {
   struct vk_buffer vk;

   struct wrapper_device *device;
   struct list_head link;
   VkBuffer dispatch_handle;
   VkDeviceSize size;
   VkDeviceSize offset;
   void *mapped_address;
   int is_mapped;
   VkDeviceMemory memory;
   struct wrapper_command_buffer *wcb;
   VkImage bcn_image;                    /* transient BCn image freed with the command buffer */
   VkExternalMemoryHandleTypeFlags handle_types;
};

struct wrapper_buffer *
get_wrapper_buffer_from_handle_locked(struct wrapper_device *device, VkBuffer buffer);

struct wrapper_buffer *
get_wrapper_buffer_from_handle(struct wrapper_device *device, VkBuffer buffer);

#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EMULATED_B8G8R8A8_CREATE_INFO_EXT 1000701234
typedef struct VkEmulatedB8G8R8A8CreateInfoExt {
    VkStructureType          sType;
    const void*              pNext;
} VkEmulatedB8G8R8A8CreateInfoExt;

struct wrapper_image {
   struct vk_image vk;

   struct wrapper_device *device;
   struct list_head link;
   VkImage dispatch_handle;
   VkImageCreateInfo info;

   bool is_emulated_bgra8;
   bool is_wsi_image;
   VkExternalMemoryHandleTypeFlags handle_types;
   uint32_t mip_drop;
   uint32_t bcn_hq_jobs;                  /* background ASTC jobs for this image */
   VkImageLayout bcn_layout;              /* last layout recorded while it has some */
   struct wrapper_command_buffer *bcn_touch; /* last command buffer that moved it */
};

struct wrapper_image *
get_wrapper_image_from_handle_locked(struct wrapper_device *device, VkImage image);

struct wrapper_image *
get_wrapper_image_from_handle(struct wrapper_device *device, VkImage image);

struct wrapper_fence {
	struct vk_fence vk;

	struct wrapper_device *device;
	struct list_head link;
	VkFence dispatch_handle;
};

/* One chunked descriptor pool used to service emulated push-descriptor sets for
 * a single command buffer recording. Pools live on the command buffer and are
 * reset (not freed) at Begin/Reset -- safe because a CB can't be re-recorded
 * while still executing -- and destroyed when the command buffer is freed. */
struct wrapper_push_pool {
   struct wrapper_push_pool *next;
   VkDescriptorPool pool;
   VkDescriptorSetLayout layout;   /* the set layout this pool is sized for */
   uint32_t remaining;             /* sets left before this pool is exhausted */
};

struct wrapper_push_record {
   VkPipelineLayout layout;
   VkShaderStageFlags stages;
   uint32_t offset;
   uint32_t size;
   uint8_t data[256];
};

struct wrapper_command_buffer {
   struct vk_command_buffer vk;

   struct wrapper_device *device;
   struct list_head link;
   VkCommandPool pool;
   struct wrapper_fence *fence;
   VkCommandBuffer dispatch_handle;
   struct wrapper_push_pool *push_pools;   /* emulated push-descriptor pools */
   struct list_head dynamic_render_objects;
   bool dynamic_rendering_active;
   struct list_head bcn_transients;
   /* Compute pipeline and push constants the application set, put back after
    * a BCn shader dispatch. */
   VkPipeline compute_pipeline;
   uint32_t push_count;
   struct wrapper_push_record push[8];
};

VK_DEFINE_HANDLE_CASTS(wrapper_command_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

struct wrapper_device_memory {
   struct AHardwareBuffer *ahardware_buffer;
   struct wrapper_device *device;
   struct list_head link;
   int fd;
   void *map_address;
   size_t map_size;
   size_t alloc_size;
   VkDeviceMemory dispatch_handle;
   const VkAllocationCallbacks *alloc;
};

/* Records kept for push-descriptor emulation, keyed by the driver handle. */
struct wrapper_push_dsl {              /* per VkDescriptorSetLayout */
   bool is_push;
   VkDescriptorPoolSize sizes[16];     /* pool sizing derived from the bindings */
   uint32_t size_count;
};

struct wrapper_push_pl {               /* per VkPipelineLayout */
   uint32_t set_layout_count;
   VkDescriptorSetLayout *set_layouts; /* owned copy of the app's set layouts */
};

struct wrapper_push_template {         /* per VkDescriptorUpdateTemplate */
   bool is_push;
   VkPipelineBindPoint bind_point;
   VkPipelineLayout pipeline_layout;
   uint32_t set;
};

struct wrapper_image_view {
   VkImageView handle;
   VkImage image;
   VkFormat format;
   VkSampleCountFlagBits samples;
};

struct wrapper_dynamic_pipeline {
   VkPipeline pipeline;
   VkRenderPass render_pass;
};

struct wrapper_dynamic_render_object {
   struct list_head link;
   VkRenderPass render_pass;
   VkFramebuffer framebuffer;
};

/* An imageless VkFramebuffer the wrapper handed out itself, because the base
 * driver has no VK_KHR_imageless_framebuffer.  It holds no driver object: the
 * real framebuffer is built at vkCmdBeginRenderPass, once the image views
 * arrive in VkRenderPassAttachmentBeginInfo. */
struct wrapper_imageless_framebuffer {
   uint32_t attachment_count;
   uint32_t width;
   uint32_t height;
   uint32_t layers;
};

VkResult enumerate_physical_device(struct vk_instance *_instance);
void destroy_physical_device(struct vk_physical_device *pdevice);

void
wrapper_setup_device_features(struct wrapper_physical_device *physical_device);

uint32_t
wrapper_select_device_memory_type(struct wrapper_device *device,
                                  VkMemoryPropertyFlags flags);

VkResult
wrapper_device_memory_create(struct wrapper_device *device,
                             const VkAllocationCallbacks *alloc,
                             struct wrapper_device_memory **out_mem);

void
wrapper_device_memory_destroy(struct wrapper_device_memory *mem);

int wrapper_diag_on(void);
void wrapper_diag_append(const char *fmt, ...);

#endif
