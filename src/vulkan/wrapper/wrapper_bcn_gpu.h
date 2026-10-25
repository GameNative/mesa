#ifndef WRAPPER_BCN_GPU_H
#define WRAPPER_BCN_GPU_H

#include "wrapper_private.h"

struct wrapper_bcn_shadow {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceAddress address;
   VkDeviceSize size;
   uint32_t block_bytes;
   uint32_t levels;
   uint32_t layers;
   VkDeviceSize level_offset[16];
   uint32_t blocks_x[16];
   uint32_t blocks_y[16];
   uint32_t depth[16];
};

struct wrapper_bcn_shadow *
wrapper_bcn_shadow_get(struct wrapper_device *device, struct wrapper_image *wi);

void
wrapper_bcn_shadow_free(struct wrapper_device *device, struct wrapper_image *wi);

void
wrapper_bcn_shadow_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                          VkBuffer src, uint32_t count, const VkBufferImageCopy *regions);

bool
wrapper_bcn_shadow_readback(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                            VkBuffer dst, uint32_t count, const VkBufferImageCopy *regions);

int
wrapper_bcn_gpu_requested(void);

uint32_t
wrapper_bcn_gpu_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       VkImageLayout layout, uint32_t count, const VkBufferImageCopy *regions);

bool
wrapper_bcn_copy_image_region(struct wrapper_command_buffer *wcb,
                              struct wrapper_image *src, bool src_emulated,
                              VkImageLayout src_layout,
                              struct wrapper_image *dst, bool dst_emulated,
                              VkImageLayout dst_layout, const VkImageCopy *region);

void
wrapper_bcn_gpu_finish(struct wrapper_device *device);

void
wrapper_cmd_track_pipeline(struct wrapper_command_buffer *wcb, VkPipelineBindPoint bind_point,
                           VkPipeline pipeline);

void
wrapper_cmd_track_push(struct wrapper_command_buffer *wcb, VkPipelineLayout layout,
                       VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                       const void *data);

void
wrapper_cmd_track_reset(struct wrapper_command_buffer *wcb);

#endif
