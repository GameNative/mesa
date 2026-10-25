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

int
wrapper_bcn_gpu_path(void);

bool
wrapper_bcn_gpu_misaligned(const struct wrapper_image *wi, const VkBufferImageCopy *c);

uint32_t
wrapper_bcn_gpu_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       struct wrapper_buffer *wb, VkImageLayout layout, uint32_t count,
                       const VkBufferImageCopy *regions);

void
wrapper_bcn_gpu_init(struct wrapper_device *device, uint32_t family);

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

void
wrapper_bcn_hq_capture(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       struct wrapper_buffer *wb, VkImageLayout layout,
                       const VkBufferImageCopy *c, uint32_t block);

void
wrapper_bcn_hq_write(struct wrapper_device *device, struct wrapper_image *wi,
                     const VkImageSubresourceLayers *sub, VkOffset3D offset, VkExtent3D extent);

void
wrapper_bcn_hq_layout(struct wrapper_device *device, VkImage image, VkImageLayout layout);

void
wrapper_bcn_hq_image_destroy(struct wrapper_device *device, struct wrapper_image *wi);

void
wrapper_bcn_hq_cmd_reset(struct wrapper_command_buffer *wcb);

void
wrapper_bcn_hq_submitted(struct wrapper_queue *queue, struct wrapper_command_buffer *wcb);

void
wrapper_bcn_hq_flush(struct wrapper_queue *queue);

void
wrapper_bcn_hq_finish(struct wrapper_device *device);

#endif
