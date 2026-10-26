#ifndef WRAPPER_BCDEC_H
#define WRAPPER_BCDEC_H

#include "wrapper_private.h"

VkFormat
get_format_for_bcn(VkFormat bcn_format);

/* 4x4-class formats are stored decoded from now on (the 4x4 encoder is unusable) */
void
wrapper_bcn_store_4x4_decoded(void);

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

void
bcn_set_vendor(uint32_t vendor);

int
bcn_scan_shader(const uint32_t *code, size_t size);

void
bcn_describe_storage(char *buf, size_t size);

void
bcn_view_swizzle(VkFormat bcn_format, VkComponentMapping *m);

size_t
bcn_upload_size(VkFormat bcn_format, int w, int h);

int
get_texel_size_for_format(VkFormat format);

int
bcn_is_native(struct wrapper_physical_device *pdev, VkFormat format);

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

int
bcn_upload_enabled(void);

void
bcn_cache_note_source(void *srcBuffer, int w, int h, int src_w,
                      VkFormat format, int offset);

float
wrapper_bcn_bc7_mse(const uint8_t blk[16], const uint8_t rgba[64]);

int
wrapper_bcn_astc4_direct(VkFormat format, const uint8_t *blk, uint8_t out[16]);

int
wrapper_bcn_astc_encode_cpu(const void *blocks, int w, int h, VkFormat format, uint8_t *out,
                            size_t len, unsigned threads, float quality);

void
decompress_bcn_format(void *srcBuffer,
                      void *dstBuffer,
                      int w,
                      int h,
                      int src_w,
                      VkFormat format,
                      int offset);

#endif
