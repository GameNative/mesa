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
   bool raises_push_constant_limit;
   bool lower_int64;
   bool fold_sets;
   bool emulate_vertex_divisor;
   bool emulate_zero_divisor;
   char *resource_type;
   VkPhysicalDevice dispatch_handle;
   VkPhysicalDeviceProperties2 properties2;
   VkPhysicalDeviceDriverProperties driver_properties;
   VkPhysicalDeviceMemoryProperties memory_properties;
   uint32_t coherent_memory_type[VK_MAX_MEMORY_TYPES];
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

enum {
   BCN_SLOT_DECODE, BCN_SLOT_ASTC4, BCN_SLOT_ASTC6, BCN_SLOT_ASTC8, BCN_SLOT_HDR,
   BCN_SLOT_ENC_BC15, BCN_SLOT_ENC_BC6H, BCN_SLOT_ENC_BC7, BCN_SLOT_COUNT
};

enum { BCN_NONE, BCN_QUEUED, BCN_BUILDING, BCN_BUILT, BCN_TESTED, BCN_FAILED };

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
   struct hash_table_u64 *device_only_memory; /* allocations the host cannot map */
   struct hash_table_u64 *image_table;
   struct hash_table_u64 *image_view_table;
   struct hash_table_u64 *imageless_fb_table;
   struct hash_table_u64 *dynamic_pipeline_table;
   struct hash_table_u64 *stride_pipeline_table;
   struct hash_table_u64 *shader_code_table;
   bool emulate_dynamic_stride;
   bool emulate_eds;
   bool emulate_eds2;
   bool fold_sets;
   simple_mtx_t fold_mutex;
   struct hash_table_u64 *fold_layout_table;
   struct hash_table_u64 *fold_pl_table;
   struct hash_table_u64 *fold_set_table;
   struct fold_merge *fold_merges;
   uint32_t fold_max_pool_count;
   uint32_t fold_set_budget;
   uint32_t max_viewports;
   struct hash_table_u64 *divisor_pipeline_table;
   bool emulate_vertex_divisor;
   bool emulate_zero_divisor;
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
   bool cmdbuf_entry_barrier;
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
   int bcn_tool_state;                   /* 0 not started, 1 started, -1 off */
   uint32_t bcn_tool_family;               /* queue family of the self-test */
   pthread_t bcn_tool_thread;
   bool bcn_tool_thread_live;
   bool bcn_tool_stop;
   bool bcn_tool_exited;
   pthread_mutex_t bcn_tool_lock;
   pthread_cond_t bcn_tool_cond;
   uint32_t bcn_tool_untested;             /* a pipeline was built since the last check */
   VkPipelineCache bcn_pipeline_cache;
   VkPipelineLayout bcn_tool_layout;
   VkBuffer bcn_tables;                    /* ASTC encoder tables */
   VkDeviceMemory bcn_tables_memory;
   VkDeviceAddress bcn_tables_address;
   VkShaderModule bcn_decode_module;
   VkShaderModule bcn_encode_module;
   VkShaderModule bcn_astc_module;
   VkShaderModule bcn_astc_hdr_module;
   VkShaderModule bcn_astc_fast_module;
   VkShaderModule bcn_astc_fast_sg_module;
   VkShaderModule bcn_split_modules[2];    /* decode per format, ASTC 4x4 without the direct step */
   VkPipeline bcn_decode_fmt[11];          /* per-format decoders when the full one does not build */
   bool bcn_decode_split;
   uint32_t bcn_decode_bad;                /* format ids whose GPU decode failed the self-test */
   VkPipeline bcn_pipe[BCN_SLOT_COUNT];
   uint32_t bcn_slot_group[BCN_SLOT_COUNT]; /* blocks per workgroup edge */
   uint32_t bcn_astc_block_only;           /* slots whose workgroup encoder failed */
   bool bcn_direct;                        /* the ASTC 4x4 encoder maps BC blocks directly */
   bool bcn_bc7_mode6;                     /* BC7 readbacks use mode 6 only */
   /* indirect group counts of dispatches recorded before their pipeline's
    * self-test (no private queue): zero until the test at the next submit */
   VkBuffer bcn_gate;
   VkDeviceMemory bcn_gate_memory;
   uint32_t *bcn_gate_map;
   uint32_t bcn_gate_used;
   uint32_t bcn_gate_pending[64][6];
   uint32_t bcn_gate_pending_count;
   bool bcn_cache_warm;                    /* the pipeline cache file had data */
   bool bcn_cache_tainted;                 /* a build failed: the cache is not saved */
   VkQueue bcn_queue;                      /* where the worker runs the self-test */
   uint32_t bcn_queue_family;
   uint32_t bcn_queue_index;
   bool bcn_queue_reserved;
   bool bcn_queue_shares_query_reset;
   simple_mtx_t bcn_queue_mutex;
   int bcn_slot_state[BCN_SLOT_COUNT];
   bool bcn_slot_late[BCN_SLOT_COUNT];     /* a wait for its build timed out */
   VkPipeline bcn_retired[BCN_SLOT_COUNT]; /* dropped by the self-test */
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
   bool device_only;                     /* bound to memory the host cannot map */
   struct wrapper_command_buffer *wcb;
   VkImage bcn_image;                    /* transient BCn image freed with the command buffer */
   VkExternalMemoryHandleTypeFlags handle_types;
};

struct wrapper_buffer *
get_wrapper_buffer_from_handle_locked(struct wrapper_device *device, VkBuffer buffer);

struct wrapper_buffer *
get_wrapper_buffer_from_handle(struct wrapper_device *device, VkBuffer buffer);

bool
wrapper_memory_device_only(struct wrapper_device *device, VkDeviceMemory memory);

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
   uint32_t bcn_owner;                    /* queue family owning it after a transfer, + 1 */
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
enum wrapper_eds_field {
   WRAPPER_EDS_CULL,
   WRAPPER_EDS_FRONT,
   WRAPPER_EDS_TOPOLOGY,
   WRAPPER_EDS_DEPTH_TEST,
   WRAPPER_EDS_DEPTH_WRITE,
   WRAPPER_EDS_DEPTH_COMPARE,
   WRAPPER_EDS_DEPTH_BOUNDS,
   WRAPPER_EDS_STENCIL_TEST,
   WRAPPER_EDS_FRONT_FAIL,
   WRAPPER_EDS_FRONT_PASS,
   WRAPPER_EDS_FRONT_DEPTH_FAIL,
   WRAPPER_EDS_FRONT_COMPARE,
   WRAPPER_EDS_BACK_FAIL,
   WRAPPER_EDS_BACK_PASS,
   WRAPPER_EDS_BACK_DEPTH_FAIL,
   WRAPPER_EDS_BACK_COMPARE,
   WRAPPER_EDS_RASTER_DISCARD,
   WRAPPER_EDS_DEPTH_BIAS,
   WRAPPER_EDS_PRIM_RESTART,
   WRAPPER_EDS_VIEWPORTS,
   WRAPPER_EDS_SCISSORS,
   WRAPPER_EDS_COUNT,
};

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
   bool secondary;
   struct wrapper_stride_pipeline *stride_pipeline;
   VkPipeline stride_bound;
   uint32_t vertex_strides[32];
   uint32_t vertex_strides_valid;
   struct list_head bcn_transients;
   /* Compute pipeline and push constants the application set, put back after
    * a BCn shader dispatch. */
   VkPipeline compute_pipeline;
   uint32_t push_count;
   struct wrapper_push_record push[8];
   uint32_t eds[WRAPPER_EDS_COUNT];
   uint32_t eds_valid;
   const struct wrapper_divisor_pipeline *divisor_pipeline;
   VkBuffer vb_buffer[32];
   VkDeviceSize vb_offset[32];
   VkDeviceSize vb_size[32];
   VkDeviceSize vb_stride[32];
   uint32_t vb_valid;
   uint32_t divisor_saved_strides[32];
   uint32_t divisor_saved_valid;
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

unsigned
wrapper_mask_feature_chain(const struct wrapper_physical_device *physical_device,
                           VkPhysicalDeviceFeatures *enabled_features,
                           void *chain);

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

bool wrapper_core13_wanted(const struct wrapper_physical_device *pdevice,
                           const char *engine_name);
void wrapper_core13_init_physical_device(struct wrapper_physical_device *pdevice);
void wrapper_core13_fill_properties(struct wrapper_physical_device *pdevice,
                                    VkPhysicalDeviceProperties2 *props);
void wrapper_format_properties3_from_legacy(struct wrapper_physical_device *pdevice,
                                            VkFormat format,
                                            VkFormatProperties2 *props);
void *wrapper_core13_lower_device_create(struct wrapper_device *device,
                                         VkDeviceCreateInfo *info,
                                         const char **exts, uint32_t *ext_count);
void wrapper_core13_free_device_create(void *storage);

void wrapper_fold_init(struct wrapper_device *device);
void wrapper_fold_note_pool(struct wrapper_device *device, const VkDescriptorPoolCreateInfo *info);
void wrapper_fold_note_layout(struct wrapper_device *device,
                              const VkDescriptorSetLayoutCreateInfo *info,
                              VkDescriptorSetLayout handle);
bool wrapper_fold_create_pipeline_layout(struct wrapper_device *device,
                                         const VkPipelineLayoutCreateInfo *info,
                                         const VkAllocationCallbacks *alloc,
                                         VkPipelineLayout *out, VkResult *result);
void wrapper_fold_destroy_pipeline_layout(struct wrapper_device *device, VkPipelineLayout layout);
bool wrapper_fold_layout_folded(struct wrapper_device *device, VkPipelineLayout layout);
uint32_t wrapper_fold_set_index(struct wrapper_device *device, VkPipelineLayout layout,
                                uint32_t set);
uint32_t *wrapper_fold_remap_spirv(struct wrapper_device *device, VkPipelineLayout layout,
                                   const uint32_t *code, size_t words);
void wrapper_fold_note_sets(struct wrapper_device *device, const VkDescriptorSetAllocateInfo *info,
                            const VkDescriptorSet *sets);
void wrapper_fold_free_sets(struct wrapper_device *device, uint32_t count,
                            const VkDescriptorSet *sets);
void wrapper_fold_reset_pool(struct wrapper_device *device, VkDescriptorPool pool);
bool wrapper_fold_remap_updates(struct wrapper_device *device,
                                uint32_t *write_count, VkWriteDescriptorSet **writes,
                                uint32_t *copy_count, VkCopyDescriptorSet **copies);
bool wrapper_fold_bind(struct wrapper_command_buffer *wcb, VkPipelineBindPoint bind_point,
                       VkPipelineLayout layout, uint32_t first, uint32_t count,
                       const VkDescriptorSet *sets, uint32_t dyn_count, const uint32_t *dyn);

struct wrapper_stride_pipeline;

void wrapper_stride_note_module(struct wrapper_device *device, VkShaderModule module,
                                const uint32_t *code, size_t size);
void wrapper_stride_forget_module(struct wrapper_device *device, VkShaderModule module);
uint32_t *wrapper_stride_module_code(struct wrapper_device *device, VkShaderModule module,
                                     size_t *size);
bool wrapper_stride_wants(const struct wrapper_device *device,
                          const VkGraphicsPipelineCreateInfo *info);
VkResult wrapper_stride_create(struct wrapper_device *device, VkPipelineCache cache,
                               const VkGraphicsPipelineCreateInfo *info,
                               const VkAllocationCallbacks *alloc, VkPipeline *pipeline);
void wrapper_stride_destroy(struct wrapper_device *device, VkPipeline pipeline);
void wrapper_stride_bind(struct wrapper_command_buffer *wcb, VkPipeline pipeline);
void wrapper_stride_set(struct wrapper_command_buffer *wcb, uint32_t first, uint32_t count,
                        const VkDeviceSize *strides);
void wrapper_stride_flush(struct wrapper_command_buffer *wcb);
void wrapper_eds_set(struct wrapper_command_buffer *wcb, enum wrapper_eds_field field,
                     uint32_t value);
void wrapper_stride_reset(struct wrapper_command_buffer *wcb);
void wrapper_stride_finish(struct wrapper_device *device);

struct wrapper_divisor_pipeline;
void *wrapper_divisor_lower(struct wrapper_device *device,
                            const VkGraphicsPipelineCreateInfo *in,
                            VkGraphicsPipelineCreateInfo *out);
void wrapper_divisor_record(struct wrapper_device *device, void *lowering, VkPipeline pipeline);
void wrapper_divisor_destroy(struct wrapper_device *device, VkPipeline pipeline);
void wrapper_divisor_bind(struct wrapper_command_buffer *wcb, VkPipeline pipeline);
void wrapper_divisor_track(struct wrapper_command_buffer *wcb, uint32_t first, uint32_t count,
                           const VkBuffer *buffers, const VkDeviceSize *offsets,
                           const VkDeviceSize *sizes, const VkDeviceSize *strides);
void wrapper_divisor_reset(struct wrapper_command_buffer *wcb);
void wrapper_divisor_finish(struct wrapper_device *device);
bool wrapper_divisor_draw(struct wrapper_command_buffer *wcb, bool indexed, uint32_t count,
                          uint32_t instance_count, uint32_t first, int32_t vertex_offset,
                          uint32_t first_instance);
bool wrapper_divisor_begin_indirect(struct wrapper_command_buffer *wcb);
void wrapper_divisor_end_indirect(struct wrapper_command_buffer *wcb);

#endif
