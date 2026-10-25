#ifndef WRAPPER_BCN_GPU_H
#define WRAPPER_BCN_GPU_H

#include "wrapper_private.h"

bool
wrapper_bcn_gpu_readback(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                         VkImageLayout layout, VkBuffer dst, uint32_t count,
                         const VkBufferImageCopy *regions);

bool
wrapper_bcn_gpu_copy_image_region(struct wrapper_command_buffer *wcb,
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
