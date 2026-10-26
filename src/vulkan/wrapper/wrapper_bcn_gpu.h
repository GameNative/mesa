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
wrapper_bcn_gpu_submit(struct wrapper_queue *queue);

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
wrapper_bcn_hq_barrier(struct wrapper_command_buffer *wcb, VkImage image, VkImageLayout old_layout,
                       VkImageLayout new_layout, uint32_t src_family, uint32_t dst_family);

const char *
wrapper_bcn_hq_settings(void);

bool
wrapper_bcn_cache_on(void);

bool
wrapper_bcn_cache_load(VkFormat bc, VkFormat stored, uint32_t w, uint32_t h,
                       const uint8_t *src, size_t row_bytes, size_t stride, uint32_t rows,
                       uint8_t *out, size_t out_len);

void
wrapper_bcn_cache_store(VkFormat bc, VkFormat stored, uint32_t w, uint32_t h,
                        const uint8_t *src, size_t row_bytes, uint32_t rows,
                        const uint8_t *data, size_t len);

void
wrapper_bcn_cache_report(void);

/* CPU BCn work the application is waiting for; the background stage yields to it */
extern uint32_t wrapper_bcn_fg_busy;

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
