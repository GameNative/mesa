#include "wrapper_bcdec.h"
#include "wrapper_log.h"
#include "wrapper_util.h"

#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include "util/xxhash.h"

#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>

#define BCDEC_BC4BC5_PRECISE
#define BCDEC_IMPLEMENTATION

#include "bcdec.h"
#include "wrapper_astcenc.h"

/* Emulated BC formats are stored as ASTC no larger than the BC data, as a
 * driver with native BC would hold them. WRAPPER_BCN_ASTC=0 decodes them
 * exactly to RGBA instead. */
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

/* WRAPPER_BCN_POLICY=bc1=6x6,bc2=4x4,...,maxdim=1024
 * Per-format ASTC block (4x4, 6x6, 8x8) and a base-level cap. A key left out
 * keeps the stock behaviour (bcn_astc_edge, no cap). bc6h takes 4x4 only. */
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
      if (idx == BCN_POL_BC6H && b != 4) {
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
   return astc_enabled() != 0;
}

/* Set per device: BC6H goes to ASTC 4x4 HDR only when the device enabled it. */
void
bcn_set_astc_hdr(int on)
{
   __atomic_store_n(&bcn_astc_hdr, on, __ATOMIC_RELAXED);
}

/* ASTC block edge for a BCn format: the policy entry, else the largest block
 * that is not bigger than the BC block in bits per texel. BC1 and BC4 hold
 * 4 bits per texel, so they take 6x6 (3.56); the 8-bit formats take 4x4.
 * WRAPPER_ASTC_BLOCK=8x8 (quality=low) takes 8x8, except for BC5. */
static int
bcn_astc_edge(int idx)
{
   int b = bcn_policy_block(idx);
   if (b)
      return b;
   if (astc_block8())
      return idx == BCN_POL_BC5 ? 4 : 8;
   return idx == BCN_POL_BC1 || idx == BCN_POL_BC4 ? 6 : 4;
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

#define WRAPPER_CACHE_DIR "/data/data/app.gamenative/files/imagefs/usr/cache"

struct decompression_params {
   int block_x;
   int block_x_src;
   int block_y_count;
   int block_y_start;
   int stride;
   int texel_size;
   int w;
   int h;
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

/* An RGB half texel as E5B9G9R9 (shared exponent, 9-bit mantissas). Negative
 * values clamp to 0. */
static uint32_t
bcn_half_to_e5b9g9r9(const uint16_t rgb[3])
{
   float v[3], m = 0.0f;
   for (int c = 0; c < 3; c++) {
      _Float16 hv;
      memcpy(&hv, &rgb[c], 2);
      v[c] = (float)hv;
      if (!(v[c] > 0.0f))
         v[c] = 0.0f;
      if (v[c] > 65408.0f)
         v[c] = 65408.0f;
      m = v[c] > m ? v[c] : m;
   }
   int e = m > 0.0f ? (int)floorf(log2f(m)) : -16;
   if (e < -16)
      e = -16;
   int shared = e + 16;
   float scale = exp2f((float)(shared - 15 - 9));
   if ((int)floorf(m / scale + 0.5f) == 512) {
      shared++;
      scale *= 2.0f;
   }
   if (shared > 31)
      shared = 31;
   uint32_t out = (uint32_t)shared << 27;
   for (int c = 0; c < 3; c++) {
      int q = (int)floorf(v[c] / scale + 0.5f);
      q = q > 511 ? 511 : q;
      out |= (uint32_t)q << (9 * c);
   }
   return out;
}

static const char *
bcn_format_name(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return "bc1";
   case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: return "bc2";
   case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: return "bc3";
   case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK: return "bc4";
   case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK: return "bc5";
   case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK: return "bc6h";
   default: return "bc7";
   }
}

/* BC4/BC5 into an RGBA scratch with the layout D3D and GL sample them with:
 * BC4 as (r, 0, 0, 1), BC5 as (r, g, 0, 1). */
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
            p[0] = tmp[y * 4 + x];
            p[1] = p[2] = 0;
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

   for (int by = 0; by < params->block_y_count; by++) {
      for (int bx = 0; bx < params->block_x; bx++) {
         int pixel_x = (bx * 4);
         int pixel_y = (by + params->block_y_start) * 4;
         /* Explicit source addressing using the source row block stride, so a
          * padded bufferRowLength does not misalign subsequent rows. */
         char *src = params->src +
            ((size_t)by * params->block_x_src + bx) * block_size;

         uint8_t scratch[128];
         int pitch = params->texel_size * 4;
         int is_signed = params->format == VK_FORMAT_BC4_SNORM_BLOCK ||
                         params->format == VK_FORMAT_BC5_SNORM_BLOCK ||
                         params->format == VK_FORMAT_BC6H_SFLOAT_BLOCK;

         switch (params->format) {
            case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
            case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
               bcdec_bc1(src, scratch, pitch);
               for (int i = 0; i < 16; i++)
                  scratch[i * 4 + 3] = 255;
               break;
            case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
               bcdec_bc1(src, scratch, pitch);
               break;
            case VK_FORMAT_BC2_SRGB_BLOCK:
            case VK_FORMAT_BC2_UNORM_BLOCK:
               bcdec_bc2(src, scratch, pitch);
               break;
            case VK_FORMAT_BC3_UNORM_BLOCK:
            case VK_FORMAT_BC3_SRGB_BLOCK:
               bcdec_bc3(src, scratch, pitch);
               break;
            case VK_FORMAT_BC4_UNORM_BLOCK:
            case VK_FORMAT_BC4_SNORM_BLOCK:
               bcdec_bc4(src, scratch, pitch, is_signed);
               break;
            case VK_FORMAT_BC5_SNORM_BLOCK:
            case VK_FORMAT_BC5_UNORM_BLOCK:
               bcdec_bc5(src, scratch, pitch, is_signed);
               break;
            case VK_FORMAT_BC6H_SFLOAT_BLOCK:
            case VK_FORMAT_BC6H_UFLOAT_BLOCK: {
               uint16_t rgb[48];
               uint16_t *rgba = (uint16_t *)scratch;
               bcdec_bc6h_half(src, rgb, 12, is_signed);
               if (params->texel_size == 4) {
                  for (int i = 0; i < 16; i++)
                     ((uint32_t *)scratch)[i] = bcn_half_to_e5b9g9r9(rgb + i * 3);
                  break;
               }
               for (int i = 0; i < 16; i++) {
                  rgba[i * 4 + 0] = rgb[i * 3 + 0];
                  rgba[i * 4 + 1] = rgb[i * 3 + 1];
                  rgba[i * 4 + 2] = rgb[i * 3 + 2];
                  rgba[i * 4 + 3] = 0x3c00;
               }
               break;
            }
            case VK_FORMAT_BC7_SRGB_BLOCK:
            case VK_FORMAT_BC7_UNORM_BLOCK:
               bcdec_bc7(src, scratch, pitch);
               break;
            default:
               memset(scratch, 0, sizeof(scratch));
               break;
         }

         int cols = MIN2(4, params->w - pixel_x);
         int rows = MIN2(4, params->h - pixel_y);
         for (int y = 0; y < rows; y++)
            memcpy(dst_base + (size_t)(pixel_y + y) * params->stride +
                      (size_t)pixel_x * params->texel_size,
                   scratch + y * pitch, (size_t)cols * params->texel_size);
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
         return VK_FORMAT_R16G16B16A16_SFLOAT;
      default:
         return VK_FORMAT_R8G8B8A8_UNORM;
   }
}

/* Storage (image) format. The UNORM formats go to ASTC (BC4 as luminance,
 * BC5 as luminance + alpha, put back in place by the image view swizzle),
 * BC6H to ASTC 4x4 HDR where the device has it, else unsigned BC6H to
 * E5B9G9R9 and signed BC6H to RGBA16F. ASTC has no signed formats, so the
 * SNORM BC4/BC5 variants are decoded. */
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
         if (__atomic_load_n(&bcn_astc_hdr, __ATOMIC_RELAXED))
            return VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK;
         if (bcn_format == VK_FORMAT_BC6H_UFLOAT_BLOCK)
            return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
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

static double
bcn_stored_bits_per_texel(VkFormat img)
{
   if (is_astc_6x6(img))
      return 128.0 / 36.0;
   if (is_astc_8x8(img))
      return 2.0;
   if (is_astc_4x4(img) || is_astc_hdr_4x4(img))
      return 8.0;
   return get_texel_size_for_format(img) * 8.0;
}

static const char *
bcn_stored_name(VkFormat img)
{
   if (is_astc_6x6(img)) return "astc6x6";
   if (is_astc_8x8(img)) return "astc8x8";
   if (is_astc_hdr_4x4(img)) return "astc4x4hdr";
   if (is_astc_4x4(img)) return "astc4x4";
   switch (img) {
   case VK_FORMAT_R8_UNORM: return "r8";
   case VK_FORMAT_R8_SNORM: return "r8snorm";
   case VK_FORMAT_R8G8_UNORM: return "rg8";
   case VK_FORMAT_R8G8_SNORM: return "rg8snorm";
   case VK_FORMAT_R16G16B16A16_SFLOAT: return "rgba16f";
   case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return "e5b9g9r9";
   default: return "rgba8";
   }
}

/* Storage format and size relative to the BC data for each BC format. */
void
bcn_describe_storage(char *buf, size_t size)
{
   static const struct { VkFormat f; const char *name; } fmts[] = {
      { VK_FORMAT_BC1_RGBA_UNORM_BLOCK, "bc1" }, { VK_FORMAT_BC2_UNORM_BLOCK, "bc2" },
      { VK_FORMAT_BC3_UNORM_BLOCK, "bc3" },      { VK_FORMAT_BC4_UNORM_BLOCK, "bc4" },
      { VK_FORMAT_BC4_SNORM_BLOCK, "bc4s" },     { VK_FORMAT_BC5_UNORM_BLOCK, "bc5" },
      { VK_FORMAT_BC5_SNORM_BLOCK, "bc5s" },     { VK_FORMAT_BC6H_UFLOAT_BLOCK, "bc6h" },
      { VK_FORMAT_BC6H_SFLOAT_BLOCK, "bc6hs" },  { VK_FORMAT_BC7_UNORM_BLOCK, "bc7" },
   };
   size_t n = snprintf(buf, size, "cpus=%ld", sysconf(_SC_NPROCESSORS_CONF));
   for (unsigned i = 0; i < sizeof(fmts) / sizeof(fmts[0]) && n < size; i++) {
      VkFormat img = get_format_for_bcn(fmts[i].f);
      double bc = get_block_size(fmts[i].f) / 2.0;
      n += snprintf(buf + n, size - n, " %s=%s(%.2fx)", fmts[i].name,
                    bcn_stored_name(img), bcn_stored_bits_per_texel(img) / bc);
   }
}

/* BC4 stored as ASTC luminance and BC5 as luminance + alpha sample as
 * (L, L, L, 1) and (L, L, L, A); compose the view swizzle so they read as
 * (r, 0, 0, 1) and (r, g, 0, 1) like the BC formats. */
void
bcn_view_swizzle(VkFormat bcn_format, VkComponentMapping *m)
{
   static const VkComponentSwizzle l[4] = {
      VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ZERO,
      VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE,
   };
   static const VkComponentSwizzle la[4] = {
      VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A,
      VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE,
   };
   const VkComponentSwizzle *map;
   if (!is_astc(get_format_for_bcn(bcn_format)))
      return;
   if (bcn_format == VK_FORMAT_BC4_UNORM_BLOCK)
      map = l;
   else if (bcn_format == VK_FORMAT_BC5_UNORM_BLOCK)
      map = la;
   else
      return;
   VkComponentSwizzle *c[4] = { &m->r, &m->g, &m->b, &m->a };
   for (int i = 0; i < 4; i++) {
      VkComponentSwizzle s = *c[i] == VK_COMPONENT_SWIZZLE_IDENTITY ?
         (VkComponentSwizzle)(VK_COMPONENT_SWIZZLE_R + i) : *c[i];
      if (s >= VK_COMPONENT_SWIZZLE_R && s <= VK_COMPONENT_SWIZZLE_A)
         s = map[s - VK_COMPONENT_SWIZZLE_R];
      *c[i] = s;
   }
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
   return (size_t)w * h * get_texel_size_for_format(img);
}

int 
get_texel_size_for_format(VkFormat format) 
{
   switch (format) {
      case VK_FORMAT_R16G16B16A16_SFLOAT:
         return 8;
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
bcn_is_native(struct wrapper_physical_device *pdev, VkFormat format)
{
   if (pdev->emulate_bcn != 3 || format < VK_FORMAT_BC1_RGB_UNORM_BLOCK ||
       format > VK_FORMAT_BC7_SRGB_BLOCK)
      return 0;
   return (pdev->native_bcn_mask >> (format - VK_FORMAT_BC1_RGB_UNORM_BLOCK)) & 1;
}

int
is_emulated_bcn(struct wrapper_physical_device *pdev, VkFormat format)
{
   if (bcn_is_native(pdev, format))
      return 0;

   switch(format) {
      case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
      case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
      case VK_FORMAT_BC2_SRGB_BLOCK:
      case VK_FORMAT_BC2_UNORM_BLOCK:
      case VK_FORMAT_BC3_UNORM_BLOCK:
      case VK_FORMAT_BC3_SRGB_BLOCK:
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
/* Upload mode: larger mips keep only their .src; the server pack has them. */
#define BCN_CACHE_UPLOAD_ENTRY_MAX 1024

static const char *
bcn_cache_src_tag(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      return "bc1";
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      /* Do not reuse opaque BC1 entries written before alpha was preserved. */
      return "bc1a";
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
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return "rgba16f";
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
   if (!bcn_upload_enabled())
      return;
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

/* BC texels as the ASTC encoder takes them: RGBA8 rows of w texels, with
 * BC4 in red and BC5 in red and green, or RGBA half texels for BC6H. */
static void
bcn_decode_for_astc(const char *src, int w, int h, int block_x_src, VkFormat format,
                    uint8_t *out)
{
   int block_size = get_block_size(format);
   int hdr = format == VK_FORMAT_BC6H_UFLOAT_BLOCK || format == VK_FORMAT_BC6H_SFLOAT_BLOCK;
   int texel = hdr ? 8 : 4;
   for (int by = 0; by < (h + 3) / 4; by++) {
      for (int bx = 0; bx < (w + 3) / 4; bx++) {
         const char *blk = src + ((size_t)by * block_x_src + bx) * block_size;
         uint8_t tile[128];
         switch (format) {
         case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
         case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
            bcdec_bc1(blk, tile, 16);
            for (int i = 0; i < 16; i++)
               tile[i * 4 + 3] = 255;
            break;
         case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
         case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            bcdec_bc1(blk, tile, 16);
            break;
         case VK_FORMAT_BC2_UNORM_BLOCK:
         case VK_FORMAT_BC2_SRGB_BLOCK:
            bcdec_bc2(blk, tile, 16);
            break;
         case VK_FORMAT_BC3_UNORM_BLOCK:
         case VK_FORMAT_BC3_SRGB_BLOCK:
            bcdec_bc3(blk, tile, 16);
            break;
         case VK_FORMAT_BC4_UNORM_BLOCK:
         case VK_FORMAT_BC5_UNORM_BLOCK:
            bcn_decode_rg_rgba(format, blk, tile, 16);
            break;
         case VK_FORMAT_BC6H_UFLOAT_BLOCK:
         case VK_FORMAT_BC6H_SFLOAT_BLOCK: {
            uint16_t rgb[48];
            uint16_t *rgba = (uint16_t *)tile;
            bcdec_bc6h_half(blk, rgb, 12, format == VK_FORMAT_BC6H_SFLOAT_BLOCK);
            for (int i = 0; i < 16; i++) {
               for (int c = 0; c < 3; c++)
                  rgba[i * 4 + c] = (rgb[i * 3 + c] & 0x8000) ? 0 : rgb[i * 3 + c];
               rgba[i * 4 + 3] = 0x3c00;
            }
            break;
         }
         default:
            bcdec_bc7(blk, tile, 16);
            break;
         }
         int cols = MIN2(4, w - bx * 4), rows = MIN2(4, h - by * 4);
         for (int y = 0; y < rows; y++)
            memcpy(out + ((size_t)(by * 4 + y) * w + bx * 4) * texel, tile + y * 4 * texel,
                   (size_t)cols * texel);
      }
   }
}

static int
bcn_astc_block_edge(VkFormat img)
{
   return is_astc_6x6(img) ? 6 : is_astc_8x8(img) ? 8 : 4;
}

static enum wrapper_astc_kind
bcn_astc_kind(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return WRAPPER_ASTC_L;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return WRAPPER_ASTC_LA;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return WRAPPER_ASTC_HDR_RGB;
   default:
      return WRAPPER_ASTC_RGBA;
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

   VkFormat img = get_format_for_bcn(format);
   int astc = is_astc(img) || is_astc_hdr_4x4(img);
   int texel_size = get_texel_size_for_format(img);
   int block_size = get_block_size(format);
   int block_x = (w + 3) / 4;
   /* source row stride in blocks (bufferRowLength may pad rows wider than w) */
   int block_x_src = ((src_w > 0 ? src_w : w) + 3) / 4;
   int block_y = (h + 3) / 4;
   int stride = w * texel_size;
   int uncompressed_size = (int)bcn_upload_size(format, w, h);
   char *src = srcBuffer + offset;
   char *dst = dstBuffer;

   if (wrapper_mark_bcn && !astc) {
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


   if (astc) {
      int hdr_in = bcn_astc_kind(format) == WRAPPER_ASTC_HDR_RGB;
      uint8_t *texels = malloc((size_t)w * h * (hdr_in ? 8 : 4));
      if (texels) {
         struct timespec t0, t1, t2;
         clock_gettime(CLOCK_MONOTONIC, &t0);
         bcn_decode_for_astc(src, w, h, block_x_src, format, texels);
         clock_gettime(CLOCK_MONOTONIC, &t1);
         int block = bcn_astc_block_edge(img);
         int srgb = get_decode_format_for_bcn(format) == VK_FORMAT_R8G8B8A8_SRGB;
         if (wrapper_astcenc_compress(texels, w, h, block, bcn_astc_kind(format), srgb,
                                      (uint8_t *)dst, uncompressed_size))
            WRAPPER_LOG(error, "BCn: ASTC encode of a %dx%d %d texture failed", w, h, format);
         clock_gettime(CLOCK_MONOTONIC, &t2);
         if (w * h >= 1024 * 1024)
            wrapper_diag_append("[BCN] cpu %s %dx%d -> astc %dx%d: decode %.1f ms, encode %.1f ms\n",
                                bcn_format_name(format), w, h, block, block,
                                (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6,
                                (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) / 1e6);
         free(texels);
      }
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
      args[0].w = w;
      args[0].h = h;
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
         args[i].w = w;
         args[i].h = h;
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
      if (!bcn_upload_enabled() ||
          (w <= BCN_CACHE_UPLOAD_ENTRY_MAX && h <= BCN_CACHE_UPLOAD_ENTRY_MAX))
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

/* WRAPPER_BCN_UPLOAD=1: leave .src sidecars and needs_full_res for the server. */
int
bcn_upload_enabled(void)
{
   static int on = -1;
   if (on == -1) {
      const char *e = getenv("WRAPPER_BCN_UPLOAD");
      on = e && !strcmp(e, "1");
   }
   return on;
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
   if (!bcn_cache_enabled() || !bcn_upload_enabled() || w < 8 || h < 8)
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
   if (!bcn_upload_enabled())
      return 0;
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
