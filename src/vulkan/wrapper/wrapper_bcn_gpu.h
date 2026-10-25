#ifndef WRAPPER_BCN_GPU_H
#define WRAPPER_BCN_GPU_H

#include "wrapper_private.h"

struct wrapper_bcn_shadow {
   VkBuffer buffer;
   VkDeviceMemory memory;
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

#endif
