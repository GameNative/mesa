#ifndef WRAPPER_ASTCENC_H
#define WRAPPER_ASTCENC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum wrapper_astc_kind {
   WRAPPER_ASTC_RGBA,      /* RGBA8 texels, all four channels */
   WRAPPER_ASTC_L,         /* RGBA8 texels, red channel only, stored as luminance */
   WRAPPER_ASTC_LA,        /* RGBA8 texels, red and green, stored as luminance + alpha */
   WRAPPER_ASTC_HDR_RGB,   /* RGBA half texels, HDR colour with LDR alpha */
};

/* Compresses a w x h image to ASTC with block x block footprint into out
 * (len bytes). Returns 0 on success. */
int
wrapper_astcenc_compress(const void *texels, int w, int h, int block,
                         enum wrapper_astc_kind kind, int srgb, uint8_t *out,
                         size_t len);

/* The same on a given number of threads (0: one per core, up to 8) and at a
 * given astcenc quality (negative: WRAPPER_ASTC_PRESET, fastest by default). */
int
wrapper_astcenc_compress_threads(const void *texels, int w, int h, int block,
                                 enum wrapper_astc_kind kind, int srgb, uint8_t *out,
                                 size_t len, unsigned threads, float quality);

/* mean squared error per channel of a 4x4 LDR ASTC block against 16 RGBA8
 * texels, or a negative value when the block cannot be decoded */
float
wrapper_astc4_block_mse(const uint8_t block[16], const uint8_t rgba[64]);

#ifdef __cplusplus
}
#endif

#endif
