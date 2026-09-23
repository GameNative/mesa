#include "wrapper_bcdec.h"
#include "wrapper_log.h"
#include "wrapper_util.h"

#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include "util/xxhash.h"

#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>

#define BCDEC_BC4BC5_PRECISE
#define BCDEC_IMPLEMENTATION

#include "bcdec.h"
#include "wrapper_astc.h"

/* Transcode BCn to ASTC 4x4 (Mali-native, stays compressed) instead of
 * decoding to RGBA8 (4-8x larger). Default on; WRAPPER_BCN_ASTC=0 falls back
 * to the RGBA decode path. */
static int
astc_enabled(void)
{
   static int e = -1;
   if (e == -1)
      e = getenv("WRAPPER_BCN_ASTC") ? atoi(getenv("WRAPPER_BCN_ASTC")) : 1;
   return e;
}

/* ASTC block footprint: 0 = 4x4 (8bpp, crisp), 1 = 8x8 (2bpp, 4x smaller/softer).
 * Selected by WRAPPER_ASTC_BLOCK ("4x4" default, "8x8"). */
static int
astc_block8(void)
{
   static int b = -1;
   if (b == -1) {
      const char *e = getenv("WRAPPER_ASTC_BLOCK");
      b = (e && strstr(e, "8x8")) ? 1 : 0;
   }
   return b;
}

/* WRAPPER_BCN_POLICY=bc1=6x6,bc2=4x4,...,bc6h=4x4,maxdim=1024
 * Per-format ASTC block (4x4, 6x6, 8x8) and a base-level cap. A key left out
 * keeps the stock behaviour (WRAPPER_ASTC_BLOCK, no cap). 6x6 has an encoder
 * for bc1 only; bc6h takes 4x4 only (ASTC HDR, when the device has it). */
enum {
   BCN_POL_BC1, BCN_POL_BC2, BCN_POL_BC3, BCN_POL_BC4, BCN_POL_BC5,
   BCN_POL_BC7, BCN_POL_BC6H, BCN_POL_COUNT
};

static const char *const bcn_pol_keys[BCN_POL_COUNT] = {
   "bc1", "bc2", "bc3", "bc4", "bc5", "bc7", "bc6h"
};

static struct {
   int block[BCN_POL_COUNT]; /* 0 = stock, else 4/6/8 */
   uint32_t maxdim;          /* 0 = no cap */
   char desc[160];
} bcn_pol;

static pthread_once_t bcn_pol_once = PTHREAD_ONCE_INIT;
static int bcn_astc_hdr;

static void
bcn_policy_parse(void)
{
   const char *e = getenv("WRAPPER_BCN_POLICY");
   if (!e) {
      snprintf(bcn_pol.desc, sizeof(bcn_pol.desc), "unset");
      return;
   }
   char *copy = strdup(e), *save = NULL;
   for (char *it = copy ? strtok_r(copy, ",", &save) : NULL; it;
        it = strtok_r(NULL, ",", &save)) {
      while (*it == ' ')
         it++;
      char *eq = strchr(it, '=');
      if (!eq) {
         WRAPPER_LOG(info, "BCn policy: ignoring '%s'", it);
         continue;
      }
      *eq = 0;
      char *val = eq + 1;
      while (*val == ' ')
         val++;
      for (char *t = val + strlen(val); t > val && t[-1] == ' '; t--)
         t[-1] = 0;
      for (char *t = eq; t > it && t[-1] == ' '; t--)
         t[-1] = 0;
      if (!strcmp(it, "maxdim")) {
         char *end = NULL;
         unsigned long v = strtoul(val, &end, 10);
         if (end == val || *end || v > 65536)
            WRAPPER_LOG(info, "BCn policy: ignoring maxdim=%s", val);
         else
            bcn_pol.maxdim = (uint32_t)v;
         continue;
      }
      int idx = -1;
      for (int i = 0; i < BCN_POL_COUNT; i++)
         if (!strcmp(it, bcn_pol_keys[i]))
            idx = i;
      int b = !strcmp(val, "4x4") ? 4 : !strcmp(val, "6x6") ? 6 :
              !strcmp(val, "8x8") ? 8 : 0;
      if (idx < 0 || !b) {
         WRAPPER_LOG(info, "BCn policy: ignoring %s=%s", it, val);
         continue;
      }
      if ((b == 6 && idx != BCN_POL_BC1) ||
          ((idx == BCN_POL_BC6H || idx == BCN_POL_BC4 || idx == BCN_POL_BC5) && b != 4)) {
         WRAPPER_LOG(info, "BCn policy: no %dx%d encoder for %s, stock", b, b, it);
         continue;
      }
      bcn_pol.block[idx] = b;
   }
   free(copy);
   int n = 0;
   for (int i = 0; i < BCN_POL_COUNT; i++) {
      int b = bcn_pol.block[i];
      n += snprintf(bcn_pol.desc + n, sizeof(bcn_pol.desc) - n,
                    b ? "%s=%dx%d " : "%s=stock ", bcn_pol_keys[i], b, b);
   }
   snprintf(bcn_pol.desc + n, sizeof(bcn_pol.desc) - n, "maxdim=%u", bcn_pol.maxdim);
   WRAPPER_LOG(info, "BCn policy: %s", bcn_pol.desc);
}

const char *
bcn_policy_desc(void)
{
   pthread_once(&bcn_pol_once, bcn_policy_parse);
   return bcn_pol.desc;
}

static int
bcn_policy_block(int idx)
{
   pthread_once(&bcn_pol_once, bcn_policy_parse);
   return bcn_pol.block[idx];
}

int
bcn_policy_bc6h_hdr(void)
{
   return astc_enabled() && bcn_policy_block(BCN_POL_BC6H) == 4;
}

/* Set per device: BC6H goes to ASTC 4x4 HDR only when the device enabled it. */
void
bcn_set_astc_hdr(int on)
{
   __atomic_store_n(&bcn_astc_hdr, on, __ATOMIC_RELAXED);
}

/* ASTC block edge for a BCn format: the policy entry, else WRAPPER_ASTC_BLOCK. */
static int
bcn_astc_edge(int idx)
{
   if (idx == BCN_POL_BC4 || idx == BCN_POL_BC5)
      return 4; /* one/two-channel maps are too lossy at 8x8 */
   int b = bcn_policy_block(idx);
   return b ? b : (astc_block8() ? 8 : 4);
}

int
is_astc_4x4(VkFormat format)
{
   return format == VK_FORMAT_ASTC_4x4_UNORM_BLOCK ||
          format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
}

int
is_astc_6x6(VkFormat format)
{
   return format == VK_FORMAT_ASTC_6x6_UNORM_BLOCK ||
          format == VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
}

int
is_astc_8x8(VkFormat format)
{
   return format == VK_FORMAT_ASTC_8x8_UNORM_BLOCK ||
          format == VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
}

int
is_astc(VkFormat format)
{
   return is_astc_4x4(format) || is_astc_6x6(format) || is_astc_8x8(format);
}

int
is_astc_hdr_4x4(VkFormat format)
{
   return format == VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK;
}

static int
bcn_has_alpha(VkFormat bcn_format)
{
   switch (bcn_format) {
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK:
   case VK_FORMAT_BC3_UNORM_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK:
   case VK_FORMAT_BC7_UNORM_BLOCK:
   case VK_FORMAT_BC7_SRGB_BLOCK:
      return 1;
   default:
      /* BC1 is treated as opaque (bcdec emits alpha=255). */
      return 0;
   }
}

#define WRAPPER_CACHE_DIR "/data/data/app.gamenative/files/imagefs/usr/cache"

struct decompression_params {
   int block_x;
   int block_x_src;
   int block_y_count;
   int block_y_start;
   int stride;
   int texel_size;
   int astc;
   int astc8;
   int bc_bx;      /* BC grid width in blocks (astc8 bounds) */
   int bc_by;      /* BC grid height in blocks (astc8 bounds) */
   int astc6;
   int astc_by;    /* ASTC 6x6 grid height (astc6 bounds) */
   int w;
   int h;
   int has_alpha;
   VkFormat format;
   char *src;
   char *dst;
};

static int
get_block_size(VkFormat format) 
{
    switch(format) {
       case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
       case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
       case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
       case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
       case VK_FORMAT_BC4_UNORM_BLOCK:
       case VK_FORMAT_BC4_SNORM_BLOCK:
          return 8;
       default:
          return 16;
    }
}

/* BC4/BC5 into an RGBA scratch with the channel layout the shaders expect
 * without a view swizzle: BC4 replicates its value over RGB (A=255), BC5 keeps
 * R/G from the block with B=0, A=255. */
static void
bcn_decode_rg_rgba(VkFormat format, const char *src, unsigned char *dst, int stride)
{
   unsigned char tmp[32];

   if (format == VK_FORMAT_BC5_UNORM_BLOCK) {
      bcdec_bc5(src, tmp, 8, 0);
      for (int y = 0; y < 4; y++) {
         for (int x = 0; x < 4; x++) {
            unsigned char *p = dst + y * stride + x * 4;
            p[0] = tmp[y * 8 + x * 2];
            p[1] = tmp[y * 8 + x * 2 + 1];
            p[2] = 0;
            p[3] = 255;
         }
      }
   } else {
      bcdec_bc4(src, tmp, 4, 0);
      for (int y = 0; y < 4; y++) {
         for (int x = 0; x < 4; x++) {
            unsigned char *p = dst + y * stride + x * 4;
            p[0] = p[1] = p[2] = tmp[y * 4 + x];
            p[3] = 255;
         }
      }
   }
}

static void *
decompression_routine(void *args)
{
   struct decompression_params *params = args;

   char *dst_base = params->dst;
   int block_size = get_block_size(params->format);

   /* 6x6 ASTC (BC1 only): a 3x3 group of BC blocks (12x12 texels) covers 2x2
    * ASTC blocks. BC blocks past the grid are clamped to the last one and texels
    * past the mip edge replicate the edge texel. block_y_start/count are group
    * rows. */
   if (params->astc6) {
      int gx_count = (params->block_x + 1) / 2;
      for (int gy = params->block_y_start;
           gy < params->block_y_start + params->block_y_count; gy++) {
         for (int gx = 0; gx < gx_count; gx++) {
            unsigned char scratch[12 * 48];
            for (int dy = 0; dy < 3; dy++) {
               int bcy = 3 * gy + dy;
               if (bcy >= params->bc_by)
                  bcy = params->bc_by - 1;
               for (int dx = 0; dx < 3; dx++) {
                  int bcx = 3 * gx + dx;
                  if (bcx >= params->bc_bx)
                     bcx = params->bc_bx - 1;
                  char *sblk = params->src +
                     ((size_t)bcy * params->block_x_src + bcx) * block_size;
                  bcdec_bc1(sblk, scratch + (dy * 4) * 48 + (dx * 4) * 4, 48);
               }
            }
            for (int sy = 0; sy < 2; sy++) {
               int BY = 2 * gy + sy;
               if (BY >= params->astc_by)
                  continue;
               for (int sx = 0; sx < 2; sx++) {
                  int BX = 2 * gx + sx;
                  if (BX >= params->block_x)
                     continue;
                  uint8_t texels[144];
                  for (int y = 0; y < 6; y++) {
                     int py = BY * 6 + y;
                     if (py >= params->h)
                        py = params->h - 1;
                     py -= gy * 12;
                     for (int x = 0; x < 6; x++) {
                        int px = BX * 6 + x;
                        if (px >= params->w)
                           px = params->w - 1;
                        px -= gx * 12;
                        memcpy(texels + (y * 6 + x) * 4, scratch + py * 48 + px * 4, 4);
                     }
                  }
                  unsigned char *blk = (unsigned char *)dst_base +
                     ((size_t)BY * params->block_x + BX) * 16;
                  astc_encode_block_6x6(texels, params->has_alpha, blk);
               }
            }
         }
      }
      return NULL;
   }

   /* 8x8 ASTC: one block per 2x2 group of BC blocks. Decode the 4 BC blocks into
    * an 8x8 RGBA scratch (row stride 32 bytes), then encode one ASTC 8x8 block. */
   if (params->astc8) {
      for (int by = 0; by < params->block_y_count; by++) {
         int BY = params->block_y_start + by;
         for (int BX = 0; BX < params->block_x; BX++) {
            unsigned char scratch[256];
            memset(scratch, 0, sizeof(scratch));
            for (int dy = 0; dy < 2; dy++) {
               for (int dx = 0; dx < 2; dx++) {
                  int bcx = 2 * BX + dx, bcy = 2 * BY + dy;
                  if (bcx >= params->bc_bx || bcy >= params->bc_by)
                     continue;
                  char *sblk = params->src +
                     ((size_t)bcy * params->block_x_src + bcx) * block_size;
                  unsigned char *dsub = scratch + (dy * 4) * 32 + (dx * 4) * 4;
                  switch (params->format) {
                  case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
                  case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
                  case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
                  case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
                     bcdec_bc1(sblk, dsub, 32); break;
                  case VK_FORMAT_BC2_SRGB_BLOCK:
                  case VK_FORMAT_BC2_UNORM_BLOCK:
                     bcdec_bc2(sblk, dsub, 32); break;
                  case VK_FORMAT_BC3_UNORM_BLOCK:
                  case VK_FORMAT_BC3_SRGB_BLOCK:
                     bcdec_bc3(sblk, dsub, 32); break;
                  case VK_FORMAT_BC7_SRGB_BLOCK:
                  case VK_FORMAT_BC7_UNORM_BLOCK:
                     bcdec_bc7(sblk, dsub, 32); break;
                  case VK_FORMAT_BC4_UNORM_BLOCK:
                  case VK_FORMAT_BC5_UNORM_BLOCK:
                     bcn_decode_rg_rgba(params->format, sblk, dsub, 32); break;
                  default: break;
                  }
               }
            }
            unsigned char *blk = (unsigned char *)dst_base +
               ((size_t)BY * params->block_x + BX) * 16;
            if (params->format == VK_FORMAT_BC5_UNORM_BLOCK)
               astc_encode_rg_8x8(scratch, blk);
            else
               astc_encode_block_8x8(scratch, params->has_alpha, blk);
         }
      }
      return NULL;
   }

   for (int by = 0; by < params->block_y_count; by++) {
      for (int bx = 0; bx < params->block_x; bx++) {
         int pixel_x = (bx * 4);
         int pixel_y = (by + params->block_y_start) * 4;
         /* Explicit source addressing using the source row block stride, so a
          * padded bufferRowLength does not misalign subsequent rows. */
         char *src = params->src +
            ((size_t)by * params->block_x_src + bx) * block_size;

         /* ASTC target: decode the BCn block into a 4x4 RGBA scratch, then
          * re-encode it as one ASTC 4x4 block written to the block grid. */
         if (params->astc) {
            uint8_t scratch[64];
            switch (params->format) {
            case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
            case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
               bcdec_bc1(src, scratch, 16);
               break;
            case VK_FORMAT_BC2_SRGB_BLOCK:
            case VK_FORMAT_BC2_UNORM_BLOCK:
               bcdec_bc2(src, scratch, 16);
               break;
            case VK_FORMAT_BC3_UNORM_BLOCK:
            case VK_FORMAT_BC3_SRGB_BLOCK:
               bcdec_bc3(src, scratch, 16);
               break;
            case VK_FORMAT_BC7_SRGB_BLOCK:
            case VK_FORMAT_BC7_UNORM_BLOCK:
               bcdec_bc7(src, scratch, 16);
               break;
            case VK_FORMAT_BC4_UNORM_BLOCK:
            case VK_FORMAT_BC5_UNORM_BLOCK:
               bcn_decode_rg_rgba(params->format, src, scratch, 16);
               break;
            default:
               break;
            }
            int block_index = (by + params->block_y_start) * params->block_x + bx;
            uint8_t *blk = (uint8_t *)dst_base + (size_t)block_index * 16;
            if (params->format == VK_FORMAT_BC5_UNORM_BLOCK)
               astc_encode_rg_4x4(scratch, blk);
            else
               astc_encode_block_4x4(scratch, params->has_alpha, blk);
            continue;
         }

         char *dst = dst_base + (pixel_y * params->stride) + (pixel_x * params->texel_size);
         if (!dst || !src)
            return NULL;

         switch (params->format) {
            case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
            case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
               bcdec_bc1(src, dst, params->stride);
               break;
            case VK_FORMAT_BC2_SRGB_BLOCK:
            case VK_FORMAT_BC2_UNORM_BLOCK:
               bcdec_bc2(src, dst, params->stride);
               break;
            case VK_FORMAT_BC3_UNORM_BLOCK:
            case VK_FORMAT_BC3_SRGB_BLOCK:
               bcdec_bc3(src, dst, params->stride);
               break;
            case VK_FORMAT_BC4_UNORM_BLOCK:
            case VK_FORMAT_BC4_SNORM_BLOCK:
               bcdec_bc4(src, dst, params->stride, params->format == VK_FORMAT_BC4_SNORM_BLOCK);
               break;
            case VK_FORMAT_BC5_SNORM_BLOCK:
            case VK_FORMAT_BC5_UNORM_BLOCK:
               bcdec_bc5(src, dst, params->stride, params->format == VK_FORMAT_BC5_SNORM_BLOCK);
               break;
            case VK_FORMAT_BC6H_SFLOAT_BLOCK:
            case VK_FORMAT_BC6H_UFLOAT_BLOCK:
               bcdec_bc6h_half(src, dst, (params->stride / params->texel_size) * 3, params->format == VK_FORMAT_BC6H_SFLOAT_BLOCK);
               break;
            case VK_FORMAT_BC7_SRGB_BLOCK:
            case VK_FORMAT_BC7_UNORM_BLOCK:
               bcdec_bc7(src, dst, params->stride);
               break;
            default:
               break;
         }
      }
   }
   
   return NULL;
}

VkFormat 
get_decode_format_for_bcn(VkFormat bcn_format)
{
   switch(bcn_format) {
      case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      case VK_FORMAT_BC2_SRGB_BLOCK:
      case VK_FORMAT_BC3_SRGB_BLOCK:
      case VK_FORMAT_BC7_SRGB_BLOCK:
         return VK_FORMAT_R8G8B8A8_SRGB;
      case VK_FORMAT_BC4_UNORM_BLOCK:
         return VK_FORMAT_R8_UNORM;
      case VK_FORMAT_BC4_SNORM_BLOCK:
         return VK_FORMAT_R8_SNORM;
      case VK_FORMAT_BC5_UNORM_BLOCK:
          return VK_FORMAT_R8G8_UNORM;
      case VK_FORMAT_BC5_SNORM_BLOCK:
         return VK_FORMAT_R8G8_SNORM;
      case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      case VK_FORMAT_BC6H_UFLOAT_BLOCK:
         return VK_FORMAT_R16G16B16_SFLOAT;
      default:
         return VK_FORMAT_R8G8B8A8_UNORM;
   }
}

/* Storage (image) format. For BC1/2/3/4/5/7 this is ASTC 4x4 (kept compressed);
 * the SNORM BC4/5 variants and BC6H stay uncompressed (decoded). */
VkFormat
get_format_for_bcn(VkFormat bcn_format)
{
   if (astc_enabled()) {
      int idx = -1;
      switch (bcn_format) {
      case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
      case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
         idx = BCN_POL_BC1; break;
      case VK_FORMAT_BC2_SRGB_BLOCK:
      case VK_FORMAT_BC2_UNORM_BLOCK:
         idx = BCN_POL_BC2; break;
      case VK_FORMAT_BC3_SRGB_BLOCK:
      case VK_FORMAT_BC3_UNORM_BLOCK:
         idx = BCN_POL_BC3; break;
      case VK_FORMAT_BC7_SRGB_BLOCK:
      case VK_FORMAT_BC7_UNORM_BLOCK:
         idx = BCN_POL_BC7; break;
      case VK_FORMAT_BC4_UNORM_BLOCK:
         idx = BCN_POL_BC4; break;
      case VK_FORMAT_BC5_UNORM_BLOCK:
         idx = BCN_POL_BC5; break;
      case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      case VK_FORMAT_BC6H_SFLOAT_BLOCK:
         if (__atomic_load_n(&bcn_astc_hdr, __ATOMIC_RELAXED) && bcn_policy_bc6h_hdr())
            return VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK;
         break;
      default:
         break;
      }
      if (idx >= 0) {
         int srgb = get_decode_format_for_bcn(bcn_format) == VK_FORMAT_R8G8B8A8_SRGB;
         int b = bcn_astc_edge(idx);
         if (b == 6)
            return srgb ? VK_FORMAT_ASTC_6x6_SRGB_BLOCK : VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
         if (b == 8)
            return srgb ? VK_FORMAT_ASTC_8x8_SRGB_BLOCK : VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
         return srgb ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK : VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
      }
   }
   return get_decode_format_for_bcn(bcn_format);
}

/* Bytes needed to hold the transcoded upload for a w*h mip of this BCn format:
 * ASTC block bytes for ASTC targets, else linear decoded size. */
size_t
bcn_upload_size(VkFormat bcn_format, int w, int h)
{
   VkFormat img = get_format_for_bcn(bcn_format);
   if (is_astc_6x6(img))
      return (size_t)((w + 5) / 6) * ((h + 5) / 6) * 16;
   if (is_astc_8x8(img))
      return (size_t)((w + 7) / 8) * ((h + 7) / 8) * 16;
   if (is_astc_4x4(img) || is_astc_hdr_4x4(img))
      return (size_t)((w + 3) / 4) * ((h + 3) / 4) * 16;
   return (size_t)w * h *
      get_texel_size_for_format(get_decode_format_for_bcn(bcn_format));
}

int 
get_texel_size_for_format(VkFormat format) 
{
   switch (format) {
      case VK_FORMAT_R16G16B16_SFLOAT:
         return 6;
      case VK_FORMAT_R8G8_UNORM:
      case VK_FORMAT_R8G8_SNORM:
         return 2;
      case VK_FORMAT_R8_UNORM:
      case VK_FORMAT_R8_SNORM:
         return 1;
      default:
         return 4;
   }
}

int
is_emulated_bcn(struct wrapper_physical_device *pdev, VkFormat format)
{
   switch(format) {
      case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
      case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
      case VK_FORMAT_BC2_SRGB_BLOCK:
      case VK_FORMAT_BC2_UNORM_BLOCK:
      case VK_FORMAT_BC3_UNORM_BLOCK:
      case VK_FORMAT_BC3_SRGB_BLOCK:
         if (pdev->emulate_bcn == 3 && 
             pdev->driver_properties.driverID == VK_DRIVER_ID_SAMSUNG_PROPRIETARY)
         {
            return 0;
         }
         else if (pdev->emulate_bcn > 1) {
            return 1;
         } else {
            return 0;
         }
         break;
      case VK_FORMAT_BC4_UNORM_BLOCK:
      case VK_FORMAT_BC4_SNORM_BLOCK:
      case VK_FORMAT_BC5_SNORM_BLOCK:
      case VK_FORMAT_BC5_UNORM_BLOCK:
      case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      case VK_FORMAT_BC6H_UFLOAT_BLOCK: 
      case VK_FORMAT_BC7_SRGB_BLOCK:
      case VK_FORMAT_BC7_UNORM_BLOCK:
         if (pdev->emulate_bcn > 1)
            return 1;
         else
            return 0;
         break;
      default:
         return 0;
   }
}

/* Disk cache entries are named by content only, so an entry produced on any
 * device (or by a server) matches on every other one:
 *   <src>_<w>x<h>_<xxh64 of the logical block rows>.<out>
 * <src> folds sRGB/UNORM (the transcode output is identical), <out> is the
 * storage format the wrapper picked (a4 = ASTC 4x4, a6 = ASTC 6x6, a8 = ASTC 8x8,
 * h4 = ASTC 4x4 HDR, rgba8 ...).
 * The hash walks the logical rows only, so a padded bufferRowLength yields the
 * same key as a tightly packed upload of the same mip. */
#define BCN_CACHE_HASH_SEED 1

static const char *
bcn_cache_src_tag(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      return "bc1";
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK:
      return "bc2";
   case VK_FORMAT_BC3_UNORM_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK:
      return "bc3";
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return "bc4";
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return "bc4s";
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return "bc5";
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return "bc5s";
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      return "bc6hu";
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return "bc6hs";
   case VK_FORMAT_BC7_UNORM_BLOCK:
   case VK_FORMAT_BC7_SRGB_BLOCK:
      return "bc7";
   default:
      return NULL;
   }
}

static const char *
bcn_cache_out_tag(VkFormat image_format)
{
   if (is_astc_4x4(image_format))
      return "a4";
   if (is_astc_6x6(image_format))
      return "a6";
   if (is_astc_8x8(image_format))
      return "a8";
   if (is_astc_hdr_4x4(image_format))
      return "h4";
   switch (image_format) {
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      return "rgba8";
   case VK_FORMAT_R8G8_UNORM:
      return "rg8";
   case VK_FORMAT_R8G8_SNORM:
      return "rg8s";
   case VK_FORMAT_R8_UNORM:
      return "r8";
   case VK_FORMAT_R8_SNORM:
      return "r8s";
   case VK_FORMAT_R16G16B16_SFLOAT:
      return "rgb16f";
   default:
      return NULL;
   }
}

static uint64_t
bcn_cache_hash(const char *src, int block_x, int block_y, int block_x_src,
               int block_size)
{
   XXH64_state_t *state = XXH64_createState();
   if (!state)
      return 0;
   XXH64_reset(state, BCN_CACHE_HASH_SEED);
   size_t row = (size_t)block_x * block_size;
   size_t stride = (size_t)block_x_src * block_size;
   for (int by = 0; by < block_y; by++)
      XXH64_update(state, src + (size_t)by * stride, row);
   uint64_t hash = XXH64_digest(state);
   XXH64_freeState(state);
   return hash;
}

static char *
bcn_cache_filename(const char *dir, VkFormat src_format, VkFormat image_format,
                   int w, int h, const char *src, int block_x, int block_y,
                   int block_x_src, int block_size)
{
   const char *src_tag = bcn_cache_src_tag(src_format);
   const char *out_tag = bcn_cache_out_tag(image_format);
   if (!src_tag || !out_tag)
      return NULL;
   uint64_t hash = bcn_cache_hash(src, block_x, block_y, block_x_src, block_size);
   char *name = NULL;
   if (asprintf(&name, "%s/%s_%dx%d_%016llx.%s", dir, src_tag, w, h,
                (unsigned long long)hash, out_tag) < 0)
      return NULL;
   return name;
}

/* Server-made entries arrive zstd compressed. libzstd ships in the imagefs,
 * so resolve it lazily; without it compressed entries are simply misses. */
typedef size_t (*bcn_zstd_decompress_t)(void *, size_t, const void *, size_t);
typedef unsigned (*bcn_zstd_is_error_t)(size_t);
typedef unsigned long long (*bcn_zstd_frame_size_t)(const void *, size_t);

static bcn_zstd_decompress_t bcn_zstd_decompress;
static bcn_zstd_is_error_t bcn_zstd_is_error;
static bcn_zstd_frame_size_t bcn_zstd_frame_size;

static int
bcn_zstd_load(void)
{
   static int loaded = -1;
   if (loaded == -1) {
      void *lib = dlopen("libzstd.so.1", RTLD_LAZY);
      if (!lib)
         lib = dlopen("libzstd.so", RTLD_LAZY);
      if (lib) {
         bcn_zstd_decompress = (bcn_zstd_decompress_t)dlsym(lib, "ZSTD_decompress");
         bcn_zstd_is_error = (bcn_zstd_is_error_t)dlsym(lib, "ZSTD_isError");
         bcn_zstd_frame_size = (bcn_zstd_frame_size_t)dlsym(lib, "ZSTD_getFrameContentSize");
      }
      loaded = (bcn_zstd_decompress && bcn_zstd_is_error) ? 1 : 0;
      if (!loaded)
         WRAPPER_LOG(bcn, "No libzstd, compressed cache entries are misses");
   }
   return loaded;
}

static int
bcn_cache_read_zstd(const char *path, void *dst, size_t size, size_t file_size)
{
   if (file_size < 4 || !bcn_zstd_load())
      return 0;
   FILE *fp = fopen(path, "rb");
   if (!fp)
      return 0;
   unsigned char *buf = malloc(file_size);
   if (!buf) {
      fclose(fp);
      return 0;
   }
   size_t length = fread(buf, 1, file_size, fp);
   fclose(fp);
   int ok = 0;
   if (length == file_size && buf[0] == 0x28 && buf[1] == 0xB5 &&
       buf[2] == 0x2F && buf[3] == 0xFD) {
      unsigned long long content = bcn_zstd_frame_size ?
         bcn_zstd_frame_size(buf, length) : (unsigned long long)size;
      if (content == (unsigned long long)size ||
          content == ~0ULL || content == ~0ULL - 1) {
         size_t out = bcn_zstd_decompress(dst, size, buf, length);
         ok = !bcn_zstd_is_error(out) && out == size;
      }
      if (!ok)
         WRAPPER_LOG(bcn, "Failed to decompress texture %s from cache", path);
   }
   free(buf);
   return ok;
}

static int
bcn_cache_read(const char *path, void *dst, size_t size, int *raw)
{
   struct stat sb;
   *raw = 0;
   if (stat(path, &sb) != 0)
      return 0;
   FILE *fp = fopen(path, "rb");
   if (!fp)
      return 0;
   /* The zstd magic decides, not the size: a frame can be exactly raw size. */
   unsigned char magic[4];
   if (fread(magic, 1, 4, fp) == 4 && magic[0] == 0x28 && magic[1] == 0xB5 &&
       magic[2] == 0x2F && magic[3] == 0xFD) {
      fclose(fp);
      return bcn_cache_read_zstd(path, dst, size, (size_t)sb.st_size);
   }
   if ((size_t)sb.st_size != size) {
      fclose(fp);
      return 0;
   }
   *raw = 1;
   rewind(fp);
   size_t length = fread(dst, 1, size, fp);
   fclose(fp);
   if (length != size) {
      unlink(path);
      return 0;
   }
   return 1;
}

static void
bcn_cache_write(const char *path, const void *data, size_t size)
{
   char *tmp = NULL;
   if (asprintf(&tmp, "%s.%d.tmp", path, (int)getpid()) < 0)
      return;
   FILE *fp = fopen(tmp, "wb");
   if (!fp) {
      free(tmp);
      return;
   }
   size_t length = fwrite(data, 1, size, fp);
   fclose(fp);
   if (length == size && rename(tmp, path) == 0)
      WRAPPER_LOG(bcn, "Saved texture %s to cache", path);
   else {
      WRAPPER_LOG(bcn, "Failed to save texture %s to cache", path);
      unlink(tmp);
   }
   free(tmp);
}

/* Beside a device-made entry, keep the tightly packed source mip as
 * <key>.src so the app can hand it to the server for a better encode and
 * replace this entry later. */
static void
bcn_cache_write_source(const char *entry_path, const char *src, int block_x,
                       int block_y, int block_x_src, int block_size)
{
   const char *dot = strrchr(entry_path, '.');
   if (!dot || (strcmp(dot + 1, "a4") != 0 && strcmp(dot + 1, "a6") != 0 &&
                strcmp(dot + 1, "a8") != 0 && strcmp(dot + 1, "h4") != 0))
      return;
   size_t stem = (size_t)(dot - entry_path);
   char *path = malloc(stem + 5);
   if (!path)
      return;
   memcpy(path, entry_path, stem);
   memcpy(path + stem, ".src", 5);
   size_t row = (size_t)block_x * block_size;
   size_t stride = (size_t)block_x_src * block_size;
   if (stride == row) {
      bcn_cache_write(path, src, row * block_y);
   } else {
      char *tight = malloc(row * block_y);
      if (tight) {
         for (int by = 0; by < block_y; by++)
            memcpy(tight + (size_t)by * row, src + (size_t)by * stride, row);
         bcn_cache_write(path, tight, row * block_y);
         free(tight);
      }
   }
   free(path);
}

static int
bcn_cache_source_exists(const char *entry_path)
{
   const char *dot = strrchr(entry_path, '.');
   if (!dot)
      return 1;
   size_t stem = (size_t)(dot - entry_path);
   char *path = malloc(stem + 5);
   if (!path)
      return 1;
   memcpy(path, entry_path, stem);
   memcpy(path + stem, ".src", 5);
   int exists = access(path, F_OK) == 0;
   free(path);
   return exists;
}

/* BC6H -> ASTC 4x4 HDR fallback when no server entry exists: each block
 * becomes a void-extent block holding the mean of its in-bounds texels.
 * Khronos Data Format Spec, ASTC "Void-Extent Blocks" (2D):
 *   bits [8:0]     0x1FC  void-extent block mode
 *   bit  9         1      HDR, the colour is FP16 (0 = UNORM16)
 *   bits [11:10]   0b11   reserved
 *   bits [63:12]   all 1  min/max S and T (13 bits each), no extent
 *   bits [127:64]  R, G, B, A as 16-bit values, R lowest
 * ASTC HDR has no negative values, so SFLOAT texels clamp at 0. */
static void
bcn_encode_bc6h_void(const char *src, uint8_t *dst, int w, int h,
                     int block_x, int block_y, int block_x_src, int is_signed)
{
   static const uint8_t head[8] = { 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
   for (int by = 0; by < block_y; by++) {
      for (int bx = 0; bx < block_x; bx++) {
         unsigned short half[48];
         bcdec_bc6h_half(src + ((size_t)by * block_x_src + bx) * 16, half, 12,
                         is_signed);
         float sum[3] = { 0, 0, 0 };
         int n = 0;
         for (int y = 0; y < 4 && by * 4 + y < h; y++) {
            for (int x = 0; x < 4 && bx * 4 + x < w; x++) {
               for (int c = 0; c < 3; c++) {
                  _Float16 v;
                  memcpy(&v, &half[(y * 4 + x) * 3 + c], 2);
                  sum[c] += (float)v;
               }
               n++;
            }
         }
         uint8_t *blk = dst + ((size_t)by * block_x + bx) * 16;
         memcpy(blk, head, 8);
         for (int c = 0; c < 4; c++) {
            float f = c == 3 ? 1.0f : (n ? sum[c] / n : 0.0f);
            if (!(f > 0.0f))
               f = 0.0f;
            if (f > 65504.0f)
               f = 65504.0f;
            _Float16 hv = (_Float16)f;
            uint16_t bits;
            memcpy(&bits, &hv, 2);
            blk[8 + 2 * c] = bits & 0xFF;
            blk[9 + 2 * c] = bits >> 8;
         }
      }
   }
}

void
decompress_bcn_format(void *srcBuffer,
					  void *dstBuffer,
					  int w,
					  int h,
					  int src_w,
					  VkFormat format,
					  int offset)
{
   static int wrapper_mark_bcn =  -1;
   static int wrapper_no_bcn_thread = -1;
   static int wrapper_use_bcn_cache = -1;
   static char *wrapper_cache_path = NULL;

   if (wrapper_mark_bcn == -1)
      wrapper_mark_bcn = getenv("WRAPPER_MARK_BCN") && atoi(getenv("WRAPPER_MARK_BCN"));

   if (wrapper_no_bcn_thread == -1)
      wrapper_no_bcn_thread = getenv("WRAPPER_NO_BCN_THREAD") && atoi(getenv("WRAPPER_NO_BCN_THREAD"));

   if (wrapper_use_bcn_cache == -1)
      wrapper_use_bcn_cache = getenv("WRAPPER_USE_BCN_CACHE") ? atoi(getenv("WRAPPER_USE_BCN_CACHE")) : 1;

   if (wrapper_cache_path == NULL)
      wrapper_cache_path = getenv("WRAPPER_CACHE_PATH") ? getenv("WRAPPER_CACHE_PATH") : WRAPPER_CACHE_DIR;

   int astc = is_astc_4x4(get_format_for_bcn(format));
   int astc8 = is_astc_8x8(get_format_for_bcn(format));
   int astc6 = is_astc_6x6(get_format_for_bcn(format));
   int hdr = is_astc_hdr_4x4(get_format_for_bcn(format));
   int has_alpha = bcn_has_alpha(format);
   int texel_size = get_texel_size_for_format(get_decode_format_for_bcn(format));
   int block_size = get_block_size(format);
   int block_x = (w + 3) / 4;
   /* source row stride in blocks (bufferRowLength may pad rows wider than w) */
   int block_x_src = ((src_w > 0 ? src_w : w) + 3) / 4;
   int block_y = (h + 3) / 4;
   int block_x8 = (w + 7) / 8;
   int block_y8 = (h + 7) / 8;
   int block_x6 = (w + 5) / 6;
   int block_y6 = (h + 5) / 6;
   int stride = w * texel_size;
   int uncompressed_size = astc8 ? (block_x8 * block_y8 * 16)
                                 : (astc ? (block_x * block_y * 16) : (w * h * texel_size));
   if (astc6)
      uncompressed_size = block_x6 * block_y6 * 16;
   if (hdr)
      uncompressed_size = block_x * block_y * 16;
   char *src = srcBuffer + offset;
   char *dst = dstBuffer;

   if (wrapper_mark_bcn && !astc && !astc6 && !hdr) {
      WRAPPER_LOG(bcn, "Filling %dx%d BCn %d texture with custom color", w, h, format);

      for (int i = 0; i < h; i++) {
         for (int j = 0; j < w; j++) {
            dst = dstBuffer + (i * stride) + (j * texel_size);
            
            switch(format) {
               case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
               case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
               case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
               case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
                  /* Yellow */
                  dst[0] = 0xFF;
                  dst[1] = 0xFF;
                  dst[2] = 0;
                  dst[3] = 255;
                  break;
               case VK_FORMAT_BC2_SRGB_BLOCK:
               case VK_FORMAT_BC2_UNORM_BLOCK:
                  /* Blue */
                  dst[0] = 0;
                  dst[1] = 0;
                  dst[2] = 0xFF;
                  dst[3] = 255;
                  break;
                case VK_FORMAT_BC3_UNORM_BLOCK:
                case VK_FORMAT_BC3_SRGB_BLOCK:
                  /* Light Blue */
                  dst[0] = 0;
                  dst[1] = 0xFF;
                  dst[2] = 0xFF;
                  dst[3] = 255;
                  break;
               case VK_FORMAT_BC4_UNORM_BLOCK:
               case VK_FORMAT_BC4_SNORM_BLOCK:
                  /* Red */
                  dst[0] = 0xFF;
                  break;
               case VK_FORMAT_BC5_UNORM_BLOCK:
               case VK_FORMAT_BC5_SNORM_BLOCK:
                  /* Green */
                  dst[0] = 0;
                  dst[1] = 0xFF;
                  break;
               case VK_FORMAT_BC6H_SFLOAT_BLOCK:
               case VK_FORMAT_BC6H_UFLOAT_BLOCK:
                  /* Purple */
                  dst[0] = 0x90;
                  dst[1] = 0x40;
                  dst[2] = 0xA0;
                  break;
               case VK_FORMAT_BC7_UNORM_BLOCK:
               case VK_FORMAT_BC7_SRGB_BLOCK:
                  /* Black */
                  dst[0] = 0xFF;
                  dst[1] = 0;
                  dst[2] = 0xFF;
                  dst[3] = 255;
                  break;
               default:
                  break;
            }
         }
      }

      return;
   }

   /* Optional disk cache of the transcoded output, keyed by a hash of the
    * compressed source. Skips decode+encode on subsequent loads. Only touched
    * when explicitly enabled, so there is zero overhead by default. */
   char *cache_filename = NULL;
   if (wrapper_use_bcn_cache) {
      CREATE_FOLDER(wrapper_cache_path, 0700);
      cache_filename = bcn_cache_filename(wrapper_cache_path, format,
         get_format_for_bcn(format), w, h, src, block_x, block_y, block_x_src,
         block_size);
      int cache_raw = 0;
      if (cache_filename &&
          bcn_cache_read(cache_filename, dst, uncompressed_size, &cache_raw)) {
         WRAPPER_LOG(bcn, "Restored texture %s from cache", cache_filename);
         if (cache_raw && w >= 8 && h >= 8 &&
             !bcn_cache_source_exists(cache_filename))
            bcn_cache_write_source(cache_filename, src, block_x, block_y,
                                   block_x_src, block_size);
         free(cache_filename);
         return;
      }
   }


   if (hdr) {
      bcn_encode_bc6h_void(src, (uint8_t *)dst, w, h, block_x, block_y,
                           block_x_src, format == VK_FORMAT_BC6H_SFLOAT_BLOCK);
   } else if (astc6) {
      /* 6x6 ASTC: threaded over rows of 12x12-texel groups (2 ASTC block rows). */
      int group_y = (block_y6 + 1) / 2;
      int core_count = sysconf(_SC_NPROCESSORS_CONF);
      int num_threads = (group_y >= core_count) ? core_count : (group_y >= 4 ? 4 : 1);
      int rows_per = group_y / num_threads, rem = group_y % num_threads;
      int bc1_alpha = format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK ||
                      format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
      pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
      struct decompression_params *args = calloc(num_threads, sizeof(struct decompression_params));
      int cur = 0;
      for (int i = 0; i < num_threads; i++) {
         int rows = rows_per + ((i < rem) ? 1 : 0);
         args[i].src = src;
         args[i].dst = dst;
         args[i].block_x = block_x6;
         args[i].block_x_src = block_x_src;
         args[i].format = format;
         args[i].block_y_count = rows;
         args[i].block_y_start = cur;
         args[i].astc_by = block_y6;
         args[i].bc_bx = block_x;
         args[i].bc_by = block_y;
         args[i].w = w;
         args[i].h = h;
         args[i].astc6 = 1;
         args[i].has_alpha = bc1_alpha;
         pthread_create(&threads[i], NULL, decompression_routine, &args[i]);
         cur += rows;
      }
      for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
      free(threads);
      free(args);
   } else if (astc8) {
      /* 8x8 ASTC: each block covers a 2x2 group of BC blocks (8 = 2*4, aligned).
       * Decode the 4 BC blocks into an 8x8 RGBA scratch, then encode one ASTC
       * 8x8 block. Threaded over 8x8-block rows. */
      int core_count = sysconf(_SC_NPROCESSORS_CONF);
      int num_threads = (block_y8 >= core_count) ? core_count : (block_y8 >= 4 ? 4 : 1);
      int rows_per = block_y8 / num_threads, rem = block_y8 % num_threads;
      pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
      struct decompression_params *args = malloc(sizeof(struct decompression_params) * num_threads);
      int cur = 0;
      for (int i = 0; i < num_threads; i++) {
         int rows = rows_per + ((i < rem) ? 1 : 0);
         args[i].src = src;                 /* full src; absolute BC addressing */
         args[i].dst = dst;
         args[i].block_x = block_x8;         /* ASTC 8x8 grid width */
         args[i].block_x_src = block_x_src;  /* BC source row stride (blocks) */
         args[i].format = format;
         args[i].block_y_count = rows;
         args[i].block_y_start = cur;
         args[i].texel_size = block_size;    /* BC block bytes */
         args[i].bc_bx = block_x;
         args[i].bc_by = block_y;
         args[i].astc = 0;
         args[i].astc8 = 1;
         args[i].astc6 = 0;
         args[i].has_alpha = has_alpha;
         pthread_create(&threads[i], NULL, decompression_routine, &args[i]);
         cur += rows;
      }
      for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
      free(threads);
      free(args);
   } else if (wrapper_no_bcn_thread) {
      WRAPPER_LOG(bcn, "Decompressing %dx%d BCN %d texture from main thread",
         w, h, format);
         
      struct decompression_params args[1];
      args[0].src = src;
      args[0].dst = dst;
      args[0].block_x = block_x;
      args[0].block_x_src = block_x_src;
      args[0].format = format;
      args[0].block_y_count = block_y;
      args[0].block_y_start = 0;
      args[0].stride = stride;
      args[0].texel_size = texel_size;
      args[0].astc = astc;
      args[0].astc8 = 0;
      args[0].astc6 = 0;
      args[0].has_alpha = has_alpha;
      decompression_routine(&args[0]);
   } else {
      int core_count = sysconf(_SC_NPROCESSORS_CONF);
      int num_threads;
      if (block_y >= core_count)
         num_threads = core_count;
      else if (block_y >= 4)
         num_threads = 4;
      else
         num_threads = 1;
      
      int rows_per_thread = block_y / num_threads;
      int rem = block_y % num_threads;

      pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
      struct decompression_params *args = malloc(sizeof(struct decompression_params) * num_threads);
      int current_row = 0;

      WRAPPER_LOG(bcn, "Decompressing %dx%d BCN %d texture using %d threads",
         w, h, format, num_threads);

      for (int i = 0; i < num_threads; i++) {
         int rows = rows_per_thread + ((i < rem) ? 1 : 0);
         args[i].src = src + ((size_t)current_row * block_x_src * block_size);
         args[i].dst = dst;
         args[i].block_x = block_x;
         args[i].block_x_src = block_x_src;
         args[i].format = format;
         args[i].block_y_count = rows;
         args[i].block_y_start = current_row;
         args[i].stride = stride;
         args[i].texel_size = texel_size;
         args[i].astc = astc;
         args[i].astc8 = 0;
         args[i].astc6 = 0;
         args[i].has_alpha = has_alpha;
         pthread_create(&threads[i], NULL, decompression_routine, &args[i]);
         current_row += rows;
      }
   
      for (int i = 0; i < num_threads; i++) {
         pthread_join(threads[i], NULL);
      }

      free(threads);
      free(args);
   }

   if (wrapper_use_bcn_cache && cache_filename) {
      bcn_cache_write(cache_filename, dst, uncompressed_size);
      if (w >= 8 && h >= 8)
         bcn_cache_write_source(cache_filename, src, block_x, block_y, block_x_src, block_size);
   }

   free(cache_filename);
   
}

/* Policy maxdim: emulated BCn images whose base level exceeds it drop their
 * top mips so the stored base is <= maxdim. 0 = off. */
int
bcn_max_dim(void)
{
   pthread_once(&bcn_pol_once, bcn_policy_parse);
   return (int)bcn_pol.maxdim;
}

/* Levels to drop from an emulated BCn image (caller checks the format): 2D,
 * sampled with at most transfer src/dst besides, not sparse/aliased, and with
 * a full chain or more than mip_drop + 1 levels. */
uint32_t
bcn_cap_mip_drop(const VkImageCreateInfo *ci)
{
   uint32_t cap = (uint32_t)bcn_max_dim();
   if (!cap || ci->imageType != VK_IMAGE_TYPE_2D)
      return 0;
   if (!(ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) ||
       (ci->usage & ~(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT)))
      return 0;
   if (ci->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                    VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                    VK_IMAGE_CREATE_ALIAS_BIT))
      return 0;
   uint32_t w = ci->extent.width, h = ci->extent.height, k = 0;
   while ((w >> k) > cap || (h >> k) > cap)
      k++;
   if (!k)
      return 0;
   uint32_t full = 1;
   while (((w > h ? w : h) >> full) > 0)
      full++;
   if (ci->mipLevels < full && ci->mipLevels <= k + 1)
      return 0;
   return k;
}

/* Remap a mip range onto the capped image; false when it only covers dropped
 * levels. VK_REMAINING_MIP_LEVELS stays. */
bool
bcn_cap_range(uint32_t mip_drop, VkImageSubresourceRange *range)
{
   if (!mip_drop)
      return true;
   uint32_t base = range->baseMipLevel;
   uint32_t new_base = base > mip_drop ? base - mip_drop : 0;
   if (range->levelCount != VK_REMAINING_MIP_LEVELS) {
      uint32_t end = base + range->levelCount;
      if (end <= mip_drop)
         return false;
      range->levelCount = end - mip_drop - new_base;
   }
   range->baseMipLevel = new_base;
   return true;
}

/* Copies into dropped levels are removed, the rest move down by mip_drop.
 * Returns the number of regions kept in kept[]. */
uint32_t
bcn_cap_copy_regions(uint32_t mip_drop, const VkBufferImageCopy *regions,
                     uint32_t count, VkBufferImageCopy *kept)
{
   uint32_t n = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (regions[i].imageSubresource.mipLevel < mip_drop)
         continue;
      kept[n] = regions[i];
      kept[n].imageSubresource.mipLevel -= mip_drop;
      n++;
   }
   return n;
}

int
bcn_cache_enabled(void)
{
   static int on = -1;
   if (on == -1)
      on = getenv("WRAPPER_USE_BCN_CACHE") ? atoi(getenv("WRAPPER_USE_BCN_CACHE")) : 1;
   return on;
}

/* A dropped (capped) mip is never transcoded, but its .src still goes beside
 * its cache key so the server pack stays complete for uncapped devices. */
void
bcn_cache_note_source(void *srcBuffer, int w, int h, int src_w,
                      VkFormat format, int offset)
{
   if (!bcn_cache_enabled() || w < 8 || h < 8)
      return;
   const char *dir = getenv("WRAPPER_CACHE_PATH") ? getenv("WRAPPER_CACHE_PATH")
                                                   : WRAPPER_CACHE_DIR;
   char *src = (char *)srcBuffer + offset;
   int block_size = get_block_size(format);
   int block_x = (w + 3) / 4;
   int block_y = (h + 3) / 4;
   int block_x_src = ((src_w > 0 ? src_w : w) + 3) / 4;
   CREATE_FOLDER(dir, 0700);
   char *name = bcn_cache_filename(dir, format, get_format_for_bcn(format), w, h,
                                   src, block_x, block_y, block_x_src, block_size);
   if (name && !bcn_cache_source_exists(name))
      bcn_cache_write_source(name, src, block_x, block_y, block_x_src, block_size);
   free(name);
}

/* Cache step of the GPU transcode path, keyed exactly as decompress_bcn_format.
 * Hit: returns the entry (bcn_upload_size bytes, caller frees). Miss: leaves the
 * .src sidecar and returns NULL; the GPU result is never read back. */
void *
bcn_cache_gpu_lookup(void *srcBuffer, int w, int h, int src_w, VkFormat format,
                     int offset, size_t *size)
{
   if (!bcn_cache_enabled())
      return NULL;
   const char *dir = getenv("WRAPPER_CACHE_PATH") ? getenv("WRAPPER_CACHE_PATH")
                                                   : WRAPPER_CACHE_DIR;
   char *src = (char *)srcBuffer + offset;
   int block_size = get_block_size(format);
   int block_x = (w + 3) / 4;
   int block_y = (h + 3) / 4;
   int block_x_src = ((src_w > 0 ? src_w : w) + 3) / 4;
   CREATE_FOLDER(dir, 0700);
   char *name = bcn_cache_filename(dir, format, get_format_for_bcn(format), w, h,
                                   src, block_x, block_y, block_x_src, block_size);
   if (!name)
      return NULL;
   size_t sz = bcn_upload_size(format, w, h);
   void *dst = malloc(sz);
   int raw = 0;
   if (dst && bcn_cache_read(name, dst, sz, &raw)) {
      WRAPPER_LOG(bcn, "Restored texture %s from cache", name);
      if (raw && w >= 8 && h >= 8 && !bcn_cache_source_exists(name))
         bcn_cache_write_source(name, src, block_x, block_y, block_x_src, block_size);
      free(name);
      *size = sz;
      return dst;
   }
   free(dst);
   if (w >= 8 && h >= 8 && !bcn_cache_source_exists(name))
      bcn_cache_write_source(name, src, block_x, block_y, block_x_src, block_size);
   free(name);
   return NULL;
}

/* SPIR-V scan, independent of the Mali passes: a shader that takes the size or
 * level count of a 2D sampled image (Dim 2D, MS 0, Sampled 1) or fetches its
 * texels sees capped extents. The first one leaves <cache>/needs_full_res.
 * Returns 1 once such a shader has been seen. */
#define SPV_OP_UNDEF                  1
#define SPV_OP_TYPE_IMAGE             25
#define SPV_OP_TYPE_SAMPLED_IMAGE     27
#define SPV_OP_FUNCTION_PARAMETER     55
#define SPV_OP_LOAD                   61
#define SPV_OP_COPY_OBJECT            83
#define SPV_OP_SAMPLED_IMAGE          86
#define SPV_OP_IMAGE_FETCH            95
#define SPV_OP_IMAGE                  100
#define SPV_OP_IMAGE_QUERY_SIZE_LOD   103
#define SPV_OP_IMAGE_QUERY_SIZE       104
#define SPV_OP_IMAGE_QUERY_LEVELS     106
#define SPV_OP_SELECT                 169
#define SPV_OP_PHI                    245

static int bcn_full_res_seen;

static int
bcn_spirv_needs_full_res(const uint32_t *code, size_t words)
{
   if (words < 5 || code[0] != 0x07230203)
      return 0;
   uint32_t bound = code[3];
   if (!bound || bound > (1u << 22))
      return 0;
   uint8_t *kind = calloc(bound, 1);      /* type id: 1 = 2D sampled image */
   uint32_t *vtype = calloc(bound, 4);    /* value id -> type id */
   int hit = 0;
   if (!kind || !vtype)
      goto out;
   for (size_t i = 5; i < words && !hit;) {
      uint32_t op = code[i] & 0xFFFF, n = code[i] >> 16;
      if (!n || i + n > words)
         break;
      const uint32_t *in = code + i;
      switch (op) {
      case SPV_OP_TYPE_IMAGE:
         if (n >= 8 && in[1] < bound)
            kind[in[1]] = (in[3] == 1 && in[6] == 0 && in[7] == 1) ? 1 : 2;
         break;
      case SPV_OP_TYPE_SAMPLED_IMAGE:
         if (n >= 3 && in[1] < bound && in[2] < bound)
            kind[in[1]] = kind[in[2]];
         break;
      case SPV_OP_UNDEF:
      case SPV_OP_FUNCTION_PARAMETER:
      case SPV_OP_LOAD:
      case SPV_OP_COPY_OBJECT:
      case SPV_OP_SAMPLED_IMAGE:
      case SPV_OP_IMAGE:
      case SPV_OP_SELECT:
      case SPV_OP_PHI:
         if (n >= 3 && in[2] < bound)
            vtype[in[2]] = in[1];
         break;
      case SPV_OP_IMAGE_FETCH:
      case SPV_OP_IMAGE_QUERY_SIZE_LOD:
      case SPV_OP_IMAGE_QUERY_SIZE:
      case SPV_OP_IMAGE_QUERY_LEVELS:
         if (n >= 4 && in[3] < bound && vtype[in[3]] < bound &&
             kind[vtype[in[3]]] == 1)
            hit = 1;
         break;
      default:
         break;
      }
      i += n;
   }
out:
   free(kind);
   free(vtype);
   return hit;
}

int
bcn_scan_shader(const uint32_t *code, size_t size)
{
   if (__atomic_load_n(&bcn_full_res_seen, __ATOMIC_RELAXED))
      return 1;
   if (!code || !bcn_spirv_needs_full_res(code, size / 4))
      return 0;
   if (__atomic_exchange_n(&bcn_full_res_seen, 1, __ATOMIC_RELAXED))
      return 1;
   const char *dir = getenv("WRAPPER_CACHE_PATH") ? getenv("WRAPPER_CACHE_PATH")
                                                   : WRAPPER_CACHE_DIR;
   char *path = NULL;
   if (asprintf(&path, "%s/needs_full_res", dir) >= 0) {
      FILE *fp = fopen(path, "wb");
      if (fp)
         fclose(fp);
      WRAPPER_LOG(info, "Shader reads 2D sampled image size/texels, %s %s",
                  fp ? "marked" : "could not mark", path);
      free(path);
   }
   return 1;
}
