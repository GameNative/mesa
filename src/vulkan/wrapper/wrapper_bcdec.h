#ifndef WRAPPER_BCDEC_H
#define WRAPPER_BCDEC_H

#include "wrapper_private.h"

VkFormat
get_format_for_bcn(VkFormat bcn_format);

VkFormat
get_decode_format_for_bcn(VkFormat bcn_format);

int
is_astc_4x4(VkFormat format);

int
is_astc_6x6(VkFormat format);

int
is_astc_8x8(VkFormat format);

int
is_astc(VkFormat format);

int
is_astc_hdr_4x4(VkFormat format);

const char *
bcn_policy_desc(void);

int
bcn_policy_bc6h_hdr(void);

void
bcn_set_astc_hdr(int on);

int
bcn_scan_shader(const uint32_t *code, size_t size);

size_t
bcn_upload_size(VkFormat bcn_format, int w, int h);

int
get_texel_size_for_format(VkFormat format);

int
is_emulated_bcn(struct wrapper_physical_device *pdev, VkFormat format);

int
bcn_max_dim(void);

uint32_t
bcn_cap_mip_drop(const VkImageCreateInfo *ci);

bool
bcn_cap_range(uint32_t mip_drop, VkImageSubresourceRange *range);

uint32_t
bcn_cap_copy_regions(uint32_t mip_drop, const VkBufferImageCopy *regions,
                     uint32_t count, VkBufferImageCopy *kept);

int
bcn_cache_enabled(void);

void
bcn_cache_note_source(void *srcBuffer, int w, int h, int src_w,
                      VkFormat format, int offset);

void *
bcn_cache_gpu_lookup(void *srcBuffer, int w, int h, int src_w, VkFormat format,
                     int offset, size_t *size);

void
decompress_bcn_format(void *srcBuffer,
                      void *dstBuffer,
                      int w,
                      int h,
                      int src_w,
                      VkFormat format,
                      int offset);

#endif
