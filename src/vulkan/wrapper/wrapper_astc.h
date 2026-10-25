/*
 * Minimal ASTC 4x4 LDR encoder for transcoding BCn textures.
 *
 * Two per-block configurations, both matching the block layout parsed by
 * Mesa's _mesa_unpack_astc_2d_ldr decoder (validated offline by round-trip):
 *
 *   RGB  (opaque, has_alpha==0): single plane, CEM 8 (LDR RGB direct),
 *      8-level (3-bit) weights, pure 8-bit endpoints. block mode 0x53.
 *
 *   RGBA (has_alpha==1): DUAL plane so alpha gets its own interpolation weight
 *      (independent of RGB). CEM 12 (LDR RGBA direct), 4-level (2-bit) weights
 *      per plane, colour-component-selector = alpha. The 4x4 dual-plane bit
 *      budget forces trit-encoded endpoints (range 47); the trit ISE tables and
 *      the endpoint unquantization table below were derived from Mesa's decoder.
 *      block mode 0x442.
 *
 * Mali samples ASTC natively, so BCn textures stay ~compressed in GPU memory
 * instead of being decoded to RGBA8 (4-8x larger).
 */
#ifndef WRAPPER_ASTC_H
#define WRAPPER_ASTC_H

#include <stdint.h>
#include <string.h>

/* trit ISE encode: index t0+3t1+9t2+27t3+81t4 -> 8-bit T field (5 values). */
static const uint8_t astc_trit5[243] = {
  0,1,2,4,5,6,8,9,10,16,17,18,20,21,22,24,
  25,26,3,7,11,19,23,27,12,13,14,32,33,34,36,37,
  38,40,41,42,48,49,50,52,53,54,56,57,58,35,39,43,
  51,55,59,44,45,46,64,65,66,68,69,70,72,73,74,80,
  81,82,84,85,86,88,89,90,67,71,75,83,87,91,76,77,
  78,128,129,130,132,133,134,136,137,138,144,145,146,148,149,150,
  152,153,154,131,135,139,147,151,155,140,141,142,160,161,162,164,
  165,166,168,169,170,176,177,178,180,181,182,184,185,186,163,167,
  171,179,183,187,172,173,174,192,193,194,196,197,198,200,201,202,
  208,209,210,212,213,214,216,217,218,195,199,203,211,215,219,204,
  205,206,96,97,98,100,101,102,104,105,106,112,113,114,116,117,
  118,120,121,122,99,103,107,115,119,123,108,109,110,224,225,226,
  228,229,230,232,233,234,240,241,242,244,245,246,248,249,250,227,
  231,235,243,247,251,236,237,238,28,29,30,60,61,62,92,93,
  94,156,157,158,188,189,190,220,221,222,31,63,95,159,191,223,
  124,125,126,
};
/* trit ISE encode: index t0+3t1+9t2 -> value 0..31 (T5=T6=T7=0), partial (3 values). */
static const uint8_t astc_trit3[27] = {
  0,1,2,4,5,6,8,9,10,16,17,18,20,21,22,24,
  25,26,3,7,11,19,23,27,12,13,14,
};
/* Endpoint unquantization for range 47 (4-bit + trit): quant level -> 0..255. */
static const uint8_t astc_ce_unq[48] = {
  0,255,16,239,32,223,48,207,65,190,81,174,97,158,113,142,
  5,250,21,234,38,217,54,201,70,185,86,169,103,152,119,136,
  11,244,27,228,43,212,59,196,76,179,92,163,108,147,124,131,
};
/* Inverse of astc_ce_unq: 0..255 value -> nearest quant level (0..47). */
static const uint8_t astc_ce_quant[256] = {
  0,0,0,16,16,16,16,16,16,32,32,32,32,32,2,2,
  2,2,2,18,18,18,18,18,18,34,34,34,34,34,4,4,
  4,4,4,4,20,20,20,20,20,36,36,36,36,36,6,6,
  6,6,6,6,22,22,22,22,22,38,38,38,38,38,8,8,
  8,8,8,8,24,24,24,24,24,24,40,40,40,40,40,10,
  10,10,10,10,26,26,26,26,26,26,42,42,42,42,42,12,
  12,12,12,12,12,28,28,28,28,28,44,44,44,44,44,14,
  14,14,14,14,14,30,30,30,30,30,46,46,46,46,46,46,
  47,47,47,47,47,47,31,31,31,31,31,15,15,15,15,15,
  15,45,45,45,45,45,29,29,29,29,29,13,13,13,13,13,
  13,43,43,43,43,43,27,27,27,27,27,27,11,11,11,11,
  11,41,41,41,41,41,25,25,25,25,25,25,9,9,9,9,
  9,9,39,39,39,39,39,23,23,23,23,23,7,7,7,7,
  7,7,37,37,37,37,37,21,21,21,21,21,5,5,5,5,
  5,5,35,35,35,35,35,19,19,19,19,19,19,3,3,3,
  3,3,33,33,33,33,33,17,17,17,17,17,17,1,1,1,
};

static inline void
astc_set_bits(uint8_t *blk, int pos, int count, uint32_t value)
{
   for (int i = 0; i < count; i++) {
      if (value & (1u << i))
         blk[(pos + i) >> 3] |= 1u << ((pos + i) & 7);
   }
}

static inline uint32_t
astc_reverse_bits(uint32_t v, int count)
{
   uint32_t out = 0;
   for (int i = 0; i < count; i++)
      out |= ((v >> i) & 1u) << (count - 1 - i);
   return out;
}

/* Farthest-apart pair of texels in RGB space -> endpoint indices. BCn source
 * data lies on a line per block, so this recovers that line. */
static inline void
astc_rgb_endpoints(const uint8_t texels[64], int *ai, int *bi)
{
   long best = -1;
   *ai = 0; *bi = 0;
   for (int i = 0; i < 16; i++) {
      for (int j = i + 1; j < 16; j++) {
         long d2 = 0;
         for (int c = 0; c < 3; c++) {
            long dc = (long)texels[i * 4 + c] - texels[j * 4 + c];
            d2 += dc * dc;
         }
         if (d2 > best) { best = d2; *ai = i; *bi = j; }
      }
   }
}

/* RGB single-plane block: CEM 8, 3-bit weights, 8-bit endpoints. */
static inline void
astc_encode_rgb_4x4(const uint8_t texels[64], uint8_t out[16])
{
   memset(out, 0, 16);

   int ai, bi;
   astc_rgb_endpoints(texels, &ai, &bi);
   const uint8_t *pa = &texels[ai * 4];
   const uint8_t *pb = &texels[bi * 4];
   int sa = pa[0] + pa[1] + pa[2];
   int sb = pb[0] + pb[1] + pb[2];
   const uint8_t *lo = (sb >= sa) ? pa : pb;   /* smaller sum -> e0 (no blue contract) */
   const uint8_t *hi = (sb >= sa) ? pb : pa;
   int e0[3] = { lo[0], lo[1], lo[2] };
   int e1[3] = { hi[0], hi[1], hi[2] };

   astc_set_bits(out, 0, 11, 0x53);
   astc_set_bits(out, 13, 4, 8);
   astc_set_bits(out, 17 + 0,  8, e0[0]);
   astc_set_bits(out, 17 + 8,  8, e1[0]);
   astc_set_bits(out, 17 + 16, 8, e0[1]);
   astc_set_bits(out, 17 + 24, 8, e1[1]);
   astc_set_bits(out, 17 + 32, 8, e0[2]);
   astc_set_bits(out, 17 + 40, 8, e1[2]);

   int dir[3];
   long dlen2 = 0;
   for (int c = 0; c < 3; c++) { dir[c] = e1[c] - e0[c]; dlen2 += (long)dir[c] * dir[c]; }

   for (int t = 0; t < 16; t++) {
      int q;
      if (dlen2 == 0) {
         q = 0;
      } else {
         long dot = 0;
         for (int c = 0; c < 3; c++)
            dot += (long)(texels[t * 4 + c] - e0[c]) * dir[c];
         if (dot <= 0) q = 0;
         else if (dot >= dlen2) q = 7;
         else q = (int)((dot * 7 + dlen2 / 2) / dlen2);
      }
      astc_set_bits(out, 128 - (t + 1) * 3, 3, astc_reverse_bits((uint32_t)q, 3));
   }
}

/* Trit bit positions within a 4-bit-mantissa trit group (ce_bits==4). */
static const int astc_trit_Tpos[8] = { 4, 5, 10, 11, 16, 21, 22, 27 };
static const int astc_trit_Mpos[5] = { 0, 6, 12, 17, 23 };

/* Write 8 endpoint quant levels (0..47) as two trit groups from bit 17. */
static inline void
astc_write_trit_endpoints(uint8_t *blk, const int q[8])
{
   int base = 17;
   int t = astc_trit5[(q[0] >> 4) + 3 * (q[1] >> 4) + 9 * (q[2] >> 4) +
                      27 * (q[3] >> 4) + 81 * (q[4] >> 4)];
   for (int j = 0; j < 5; j++)
      astc_set_bits(blk, base + astc_trit_Mpos[j], 4, q[j] & 15);
   for (int j = 0; j < 8; j++)
      if ((t >> j) & 1) astc_set_bits(blk, base + astc_trit_Tpos[j], 1, 1);

   base = 45;
   int t3 = astc_trit3[(q[5] >> 4) + 3 * (q[6] >> 4) + 9 * (q[7] >> 4)];
   for (int j = 0; j < 3; j++)
      astc_set_bits(blk, base + astc_trit_Mpos[j], 4, q[5 + j] & 15);
   for (int j = 0; j < 5; j++)   /* only T0..T4 fit before the CCS at bit 62 */
      if ((t3 >> j) & 1) astc_set_bits(blk, base + astc_trit_Tpos[j], 1, 1);
}

static inline int
astc_quantize_ce(int v)
{
   if (v < 0) v = 0;
   if (v > 255) v = 255;
   return astc_ce_quant[v];
}

/* RGBA dual-plane block: CEM 12, 2-bit weights per plane, alpha on plane 1. */
static inline void
astc_encode_rgba_4x4(const uint8_t texels[64], uint8_t out[16])
{
   memset(out, 0, 16);
   astc_set_bits(out, 0, 11, 0x442);
   astc_set_bits(out, 13, 4, 12);

   int ai, bi;
   astc_rgb_endpoints(texels, &ai, &bi);
   const uint8_t *pa = &texels[ai * 4];
   const uint8_t *pb = &texels[bi * 4];
   int sa = pa[0] + pa[1] + pa[2];
   int sb = pb[0] + pb[1] + pb[2];
   const uint8_t *lo = (sb >= sa) ? pa : pb;
   const uint8_t *hi = (sb >= sa) ? pb : pa;

   int amin = 255, amax = 0;
   for (int i = 0; i < 16; i++) {
      int a = texels[i * 4 + 3];
      if (a < amin) amin = a;
      if (a > amax) amax = a;
   }
   int e0[4] = { lo[0], lo[1], lo[2], amin };
   int e1[4] = { hi[0], hi[1], hi[2], amax };

   int q[8] = {
      astc_quantize_ce(e0[0]), astc_quantize_ce(e1[0]),
      astc_quantize_ce(e0[1]), astc_quantize_ce(e1[1]),
      astc_quantize_ce(e0[2]), astc_quantize_ce(e1[2]),
      astc_quantize_ce(e0[3]), astc_quantize_ce(e1[3]),
   };

   /* Guarantee s1>=s0 on the unquantized RGB sums (quantization can flip a
    * near-tie); otherwise the decoder blue-contracts and corrupts the block. */
   int s0 = astc_ce_unq[q[0]] + astc_ce_unq[q[2]] + astc_ce_unq[q[4]];
   int s1 = astc_ce_unq[q[1]] + astc_ce_unq[q[3]] + astc_ce_unq[q[5]];
   int swap = s1 < s0;
   if (swap)
      for (int k = 0; k < 8; k += 2) { int tmp = q[k]; q[k] = q[k + 1]; q[k + 1] = tmp; }

   astc_write_trit_endpoints(out, q);
   astc_set_bits(out, 62, 2, 3);   /* CCS = alpha component */

   int dir[3];
   long dlen2 = 0;
   for (int c = 0; c < 3; c++) { dir[c] = e1[c] - e0[c]; dlen2 += (long)dir[c] * dir[c]; }
   int arange = amax - amin;

   for (int px = 0; px < 16; px++) {
      int q0;
      if (dlen2 == 0) {
         q0 = 0;
      } else {
         long dot = 0;
         for (int c = 0; c < 3; c++)
            dot += (long)(texels[px * 4 + c] - e0[c]) * dir[c];
         if (dot <= 0) q0 = 0;
         else if (dot >= dlen2) q0 = 3;
         else q0 = (int)((dot * 3 + dlen2 / 2) / dlen2);
      }
      int q1;
      if (arange == 0) {
         q1 = 0;
      } else {
         int a = texels[px * 4 + 3] - amin;
         q1 = (a * 3 + arange / 2) / arange;
         if (q1 < 0) q1 = 0;
         if (q1 > 3) q1 = 3;
      }
      if (swap) { q0 = 3 - q0; q1 = 3 - q1; }
      astc_set_bits(out, 128 - (2 * px + 1) * 2, 2, astc_reverse_bits((uint32_t)q0, 2));
      astc_set_bits(out, 128 - (2 * px + 2) * 2, 2, astc_reverse_bits((uint32_t)q1, 2));
   }
}

/* Encode a 4x4 RGBA8 block (16 texels, row-major) into a 16-byte ASTC block. */
static inline void
astc_encode_block_4x4(const uint8_t texels[64], int has_alpha, uint8_t out[16])
{
   if (has_alpha)
      astc_encode_rgba_4x4(texels, out);
   else
      astc_encode_rgb_4x4(texels, out);
}

/* ---- 8x8 blocks: same encoders but over 64 texels with a 4x4 weight grid, the
 * grid weights least-squares-fit to the per-texel ideal weights via the infill
 * pseudo-inverse (x256 fixed point). The block mode is identical to 4x4 (it
 * encodes the weight grid, not the block); ASTC_8x8 format tells the decoder to
 * infill the grid up to 64 texels. ---- */
static const int16_t astc_pinv8[16][64] = {
  {166,75,-16,-25,-4,10,4,-3,75,33,-9,-10,-3,5,2,-2,-16,-9,-2,5,-1,-1,0,0,-25,-10,5,5,0,-2,-1,0,-4,-3,-1,0,1,0,0,0,10,5,-1,-2,0,0,0,0,4,2,0,-1,0,0,0,0,-3,-2,0,0,0,0,0,0},
  {-42,45,132,93,17,-39,-14,12,-20,21,62,39,10,-19,-6,6,3,-3,-8,-15,4,1,1,0,5,-8,-21,-16,-1,6,2,-1,1,-1,-3,-1,-2,1,0,0,-2,3,7,6,1,-2,-1,0,-1,1,2,2,0,-1,0,0,1,-1,-2,-2,-1,0,0,0},
  {12,-14,-39,17,93,132,45,-42,6,-6,-19,10,39,62,21,-20,0,1,1,4,-15,-8,-3,3,-1,2,6,-1,-16,-21,-8,5,0,0,1,-2,-1,-3,-1,1,0,-1,-2,1,6,7,3,-2,0,0,-1,0,2,2,1,-1,0,0,0,-1,-2,-2,-1,1},
  {-3,4,10,-4,-25,-16,75,166,-2,2,5,-3,-10,-9,33,75,0,0,-1,-1,5,-2,-9,-16,0,-1,-2,0,5,5,-10,-25,0,0,0,1,0,-1,-3,-4,0,0,0,0,-2,-1,5,10,0,0,0,0,-1,0,2,4,0,0,0,0,0,0,-2,-3},
  {-42,-20,3,5,1,-2,-1,1,45,21,-3,-8,-1,3,1,-1,132,62,-8,-21,-3,7,2,-2,93,39,-15,-16,-1,6,2,-2,17,10,4,-1,-2,1,0,-1,-39,-19,1,6,1,-2,-1,0,-14,-6,1,2,0,-1,0,0,12,6,0,-1,0,0,0,0},
  {12,-10,-32,-23,-5,9,3,-3,-10,12,34,28,2,-9,-3,3,-32,34,99,79,10,-27,-9,9,-23,28,79,59,4,-22,-8,7,-5,2,10,4,8,-4,-1,2,9,-9,-27,-22,-4,6,2,-2,3,-3,-9,-8,-1,2,1,-1,-3,3,9,7,2,-2,-1,1},
  {-3,3,9,-5,-23,-32,-10,12,3,-3,-9,2,28,34,12,-10,9,-9,-27,10,79,99,34,-32,7,-8,-22,4,59,79,28,-23,2,-1,-4,8,4,10,2,-5,-2,2,6,-4,-22,-27,-9,9,-1,1,2,-1,-8,-9,-3,3,1,-1,-2,2,7,9,3,-3},
  {1,-1,-2,1,5,3,-20,-42,-1,1,3,-1,-8,-3,21,45,-2,2,7,-3,-21,-8,62,132,-2,2,6,-1,-16,-15,39,93,-1,0,1,-2,-1,4,10,17,0,-1,-2,1,6,1,-19,-39,0,0,-1,0,2,1,-6,-14,0,0,0,0,-1,0,6,12},
  {12,6,0,-1,0,0,0,0,-14,-6,1,2,0,-1,0,0,-39,-19,1,6,1,-2,-1,0,17,10,4,-1,-2,1,0,-1,93,39,-15,-16,-1,6,2,-2,132,62,-8,-21,-3,7,2,-2,45,21,-3,-8,-1,3,1,-1,-42,-20,3,5,1,-2,-1,1},
  {-3,3,9,7,2,-2,-1,1,3,-3,-9,-8,-1,2,1,-1,9,-9,-27,-22,-4,6,2,-2,-5,2,10,4,8,-4,-1,2,-23,28,79,59,4,-22,-8,7,-32,34,99,79,10,-27,-9,9,-10,12,34,28,2,-9,-3,3,12,-10,-32,-23,-5,9,3,-3},
  {1,-1,-2,2,7,9,3,-3,-1,1,2,-1,-8,-9,-3,3,-2,2,6,-4,-22,-27,-9,9,2,-1,-4,8,4,10,2,-5,7,-8,-22,4,59,79,28,-23,9,-9,-27,10,79,99,34,-32,3,-3,-9,2,28,34,12,-10,-3,3,9,-5,-23,-32,-10,12},
  {0,0,0,0,-1,0,6,12,0,0,-1,0,2,1,-6,-14,0,-1,-2,1,6,1,-19,-39,-1,0,1,-2,-1,4,10,17,-2,2,6,-1,-16,-15,39,93,-2,2,7,-3,-21,-8,62,132,-1,1,3,-1,-8,-3,21,45,1,-1,-2,1,5,3,-20,-42},
  {-3,-2,0,0,0,0,0,0,4,2,0,-1,0,0,0,0,10,5,-1,-2,0,0,0,0,-4,-3,-1,0,1,0,0,0,-25,-10,5,5,0,-2,-1,0,-16,-9,-2,5,-1,-1,0,0,75,33,-9,-10,-3,5,2,-2,166,75,-16,-25,-4,10,4,-3},
  {1,-1,-2,-2,-1,0,0,0,-1,1,2,2,0,-1,0,0,-2,3,7,6,1,-2,-1,0,1,-1,-3,-1,-2,1,0,0,5,-8,-21,-16,-1,6,2,-1,3,-3,-8,-15,4,1,1,0,-20,21,62,39,10,-19,-6,6,-42,45,132,93,17,-39,-14,12},
  {0,0,0,-1,-2,-2,-1,1,0,0,-1,0,2,2,1,-1,0,-1,-2,1,6,7,3,-2,0,0,1,-2,-1,-3,-1,1,-1,2,6,-1,-16,-21,-8,5,0,1,1,4,-15,-8,-3,3,6,-6,-19,10,39,62,21,-20,12,-14,-39,17,93,132,45,-42},
  {0,0,0,0,0,0,-2,-3,0,0,0,0,-1,0,2,4,0,0,0,0,-2,-1,5,10,0,0,0,1,0,-1,-3,-4,0,-1,-2,0,5,5,-10,-25,0,0,-1,-1,5,-2,-9,-16,-2,2,5,-3,-10,-9,33,75,-3,4,10,-4,-25,-16,75,166},
};

static inline int astc_round(double x) { return (int)(x >= 0 ? x + 0.5 : x - 0.5); }

/* Farthest RGB pair over 64 texels. */
static inline void
astc_rgb_endpoints64(const uint8_t texels[256], int *ai, int *bi)
{
   long best = -1;
   *ai = 0; *bi = 0;
   for (int i = 0; i < 64; i++)
      for (int j = i + 1; j < 64; j++) {
         long d2 = 0;
         for (int c = 0; c < 3; c++) {
            long dc = (long)texels[i * 4 + c] - texels[j * 4 + c];
            d2 += dc * dc;
         }
         if (d2 > best) { best = d2; *ai = i; *bi = j; }
      }
}

static inline void
astc_encode_rgb_8x8(const uint8_t texels[256], uint8_t out[16])
{
   memset(out, 0, 16);
   int ai, bi;
   astc_rgb_endpoints64(texels, &ai, &bi);
   const uint8_t *pa = &texels[ai * 4], *pb = &texels[bi * 4];
   int sa = pa[0] + pa[1] + pa[2], sb = pb[0] + pb[1] + pb[2];
   const uint8_t *lo = (sb >= sa) ? pa : pb, *hi = (sb >= sa) ? pb : pa;
   int e0[3] = { lo[0], lo[1], lo[2] }, e1[3] = { hi[0], hi[1], hi[2] };

   astc_set_bits(out, 0, 11, 0x53);
   astc_set_bits(out, 13, 4, 8);
   astc_set_bits(out, 17 + 0, 8, e0[0]);  astc_set_bits(out, 17 + 8, 8, e1[0]);
   astc_set_bits(out, 17 + 16, 8, e0[1]); astc_set_bits(out, 17 + 24, 8, e1[1]);
   astc_set_bits(out, 17 + 32, 8, e0[2]); astc_set_bits(out, 17 + 40, 8, e1[2]);

   int dir[3]; long dlen2 = 0;
   for (int c = 0; c < 3; c++) { dir[c] = e1[c] - e0[c]; dlen2 += (long)dir[c] * dir[c]; }

   double ideal[64];
   for (int t = 0; t < 64; t++) {
      double w;
      if (dlen2 == 0) w = 0;
      else {
         long dot = 0;
         for (int c = 0; c < 3; c++) dot += (long)(texels[t * 4 + c] - e0[c]) * dir[c];
         double f = (double)dot / (double)dlen2;
         if (f < 0) f = 0; if (f > 1) f = 1;
         w = f * 7.0;
      }
      ideal[t] = w;
   }
   for (int g = 0; g < 16; g++) {
      double a = 0;
      for (int t = 0; t < 64; t++) a += astc_pinv8[g][t] * ideal[t];
      int q = astc_round(a / 256.0);
      if (q < 0) q = 0; if (q > 7) q = 7;
      astc_set_bits(out, 128 - (g + 1) * 3, 3, astc_reverse_bits((uint32_t)q, 3));
   }
}

static inline void
astc_encode_rgba_8x8(const uint8_t texels[256], uint8_t out[16])
{
   memset(out, 0, 16);
   astc_set_bits(out, 0, 11, 0x442);
   astc_set_bits(out, 13, 4, 12);
   int ai, bi;
   astc_rgb_endpoints64(texels, &ai, &bi);
   const uint8_t *pa = &texels[ai * 4], *pb = &texels[bi * 4];
   int sa = pa[0] + pa[1] + pa[2], sb = pb[0] + pb[1] + pb[2];
   const uint8_t *lo = (sb >= sa) ? pa : pb, *hi = (sb >= sa) ? pb : pa;
   int amin = 255, amax = 0;
   for (int i = 0; i < 64; i++) { int a = texels[i * 4 + 3]; if (a < amin) amin = a; if (a > amax) amax = a; }
   int e0[4] = { lo[0], lo[1], lo[2], amin }, e1[4] = { hi[0], hi[1], hi[2], amax };

   int q[8] = {
      astc_quantize_ce(e0[0]), astc_quantize_ce(e1[0]),
      astc_quantize_ce(e0[1]), astc_quantize_ce(e1[1]),
      astc_quantize_ce(e0[2]), astc_quantize_ce(e1[2]),
      astc_quantize_ce(e0[3]), astc_quantize_ce(e1[3]),
   };
   int s0 = astc_ce_unq[q[0]] + astc_ce_unq[q[2]] + astc_ce_unq[q[4]];
   int s1 = astc_ce_unq[q[1]] + astc_ce_unq[q[3]] + astc_ce_unq[q[5]];
   int swap = s1 < s0;
   if (swap)
      for (int k = 0; k < 8; k += 2) { int tmp = q[k]; q[k] = q[k + 1]; q[k + 1] = tmp; }
   astc_write_trit_endpoints(out, q);
   astc_set_bits(out, 62, 2, 3);

   int dir[3]; long dlen2 = 0;
   for (int c = 0; c < 3; c++) { dir[c] = e1[c] - e0[c]; dlen2 += (long)dir[c] * dir[c]; }
   int arange = amax - amin;

   double id0[64], id1[64];
   for (int t = 0; t < 64; t++) {
      double w0;
      if (dlen2 == 0) w0 = 0;
      else {
         long dot = 0;
         for (int c = 0; c < 3; c++) dot += (long)(texels[t * 4 + c] - e0[c]) * dir[c];
         double f = (double)dot / (double)dlen2;
         if (f < 0) f = 0; if (f > 1) f = 1;
         w0 = f * 3.0;
      }
      double w1;
      if (arange == 0) w1 = 0;
      else {
         double f = (double)(texels[t * 4 + 3] - amin) / (double)arange;
         if (f < 0) f = 0; if (f > 1) f = 1;
         w1 = f * 3.0;
      }
      if (swap) { w0 = 3.0 - w0; w1 = 3.0 - w1; }
      id0[t] = w0; id1[t] = w1;
   }
   for (int g = 0; g < 16; g++) {
      double a0 = 0, a1 = 0;
      for (int t = 0; t < 64; t++) { a0 += astc_pinv8[g][t] * id0[t]; a1 += astc_pinv8[g][t] * id1[t]; }
      int q0 = astc_round(a0 / 256.0), q1 = astc_round(a1 / 256.0);
      if (q0 < 0) q0 = 0; if (q0 > 3) q0 = 3;
      if (q1 < 0) q1 = 0; if (q1 > 3) q1 = 3;
      astc_set_bits(out, 128 - (2 * g + 1) * 2, 2, astc_reverse_bits((uint32_t)q0, 2));
      astc_set_bits(out, 128 - (2 * g + 2) * 2, 2, astc_reverse_bits((uint32_t)q1, 2));
   }
}

static inline void
astc_encode_rg_8x8(const uint8_t texels[256], uint8_t out[16])
{
   memset(out, 0, 16);
   astc_set_bits(out, 0, 11, 0x442);
   astc_set_bits(out, 13, 4, 12);

   int rmin = 255, rmax = 0, gmin = 255, gmax = 0;
   for (int i = 0; i < 64; i++) {
      int r = texels[i * 4], g = texels[i * 4 + 1];
      if (r < rmin) rmin = r;
      if (r > rmax) rmax = r;
      if (g < gmin) gmin = g;
      if (g > gmax) gmax = g;
   }

   int q[8] = {
      astc_quantize_ce(rmin), astc_quantize_ce(rmax),
      astc_quantize_ce(gmin), astc_quantize_ce(gmax),
      astc_quantize_ce(0), astc_quantize_ce(0),
      astc_quantize_ce(255), astc_quantize_ce(255),
   };
   astc_write_trit_endpoints(out, q);
   astc_set_bits(out, 62, 2, 1);

   int r0 = astc_ce_unq[q[0]], r1 = astc_ce_unq[q[1]];
   int g0 = astc_ce_unq[q[2]], g1 = astc_ce_unq[q[3]];
   int rrange = r1 - r0, grange = g1 - g0;

   double id0[64], id1[64];
   for (int t = 0; t < 64; t++) {
      double w0 = 0, w1 = 0;
      if (rrange > 0) {
         double f = (double)(texels[t * 4] - r0) / (double)rrange;
         if (f < 0) f = 0; if (f > 1) f = 1;
         w0 = f * 3.0;
      }
      if (grange > 0) {
         double f = (double)(texels[t * 4 + 1] - g0) / (double)grange;
         if (f < 0) f = 0; if (f > 1) f = 1;
         w1 = f * 3.0;
      }
      id0[t] = w0; id1[t] = w1;
   }
   for (int g = 0; g < 16; g++) {
      double a0 = 0, a1 = 0;
      for (int t = 0; t < 64; t++) { a0 += astc_pinv8[g][t] * id0[t]; a1 += astc_pinv8[g][t] * id1[t]; }
      int q0 = astc_round(a0 / 256.0), q1 = astc_round(a1 / 256.0);
      if (q0 < 0) q0 = 0; if (q0 > 3) q0 = 3;
      if (q1 < 0) q1 = 0; if (q1 > 3) q1 = 3;
      astc_set_bits(out, 128 - (2 * g + 1) * 2, 2, astc_reverse_bits((uint32_t)q0, 2));
      astc_set_bits(out, 128 - (2 * g + 2) * 2, 2, astc_reverse_bits((uint32_t)q1, 2));
   }
}

/* Encode an 8x8 RGBA8 block (64 texels, row-major) into a 16-byte ASTC block. */
static inline void
astc_encode_block_8x8(const uint8_t texels[256], int has_alpha, uint8_t out[16])
{
   if (has_alpha)
      astc_encode_rgba_8x8(texels, out);
   else
      astc_encode_rgb_8x8(texels, out);
}

/* ---- 6x6 blocks (BC1 opt-in): same encoders over 36 texels with a 4x4 weight
 * grid fit via the 6x6 infill pseudo-inverse (x256 fixed point); block modes as
 * for 4x4, the ASTC_6x6 format makes the decoder infill to 36 texels. ---- */
static const int16_t astc_pinv6[16][36] = {
  {217,50,-38,2,8,-3,50,7,-9,1,2,-1,-38,-9,8,-1,-1,0,2,1,-1,0,0,0,8,2,-1,0,0,0,-3,-1,0,0,0,0},
  {-50,134,186,-11,-38,14,-13,36,46,-6,-11,4,8,-24,-36,4,6,-2,0,0,4,-1,1,0,-1,4,6,1,-1,0,1,-2,-2,0,0,0},
  {14,-38,-11,186,134,-50,4,-11,-6,46,36,-13,-2,6,4,-36,-24,8,0,1,-1,4,0,0,0,-1,1,6,4,-1,0,0,0,-2,-2,1},
  {-3,8,2,-38,50,217,-1,2,1,-9,7,50,0,-1,-1,8,-9,-38,0,0,0,-1,1,2,0,0,0,-1,2,8,0,0,0,0,-1,-3},
  {-50,-13,8,0,-1,1,134,36,-24,0,4,-2,186,46,-36,4,6,-2,-11,-6,4,-1,1,0,-38,-11,6,1,-1,0,14,4,-2,0,0,0},
  {13,-31,-44,1,8,-3,-31,77,112,0,-20,8,-44,112,173,-18,-30,12,1,0,-18,6,-4,1,8,-20,-30,-4,3,-1,-3,8,12,1,-1,0},
  {-3,8,1,-44,-31,13,8,-20,0,112,77,-31,12,-30,-18,173,112,-44,1,-4,6,-18,0,1,-1,3,-4,-30,-20,8,0,-1,1,12,8,-3},
  {1,-1,0,8,-13,-50,-2,4,0,-24,36,134,-2,6,4,-36,46,186,0,1,-1,4,-6,-11,0,-1,1,6,-11,-38,0,0,0,-2,4,14},
  {14,4,-2,0,0,0,-38,-11,6,1,-1,0,-11,-6,4,-1,1,0,186,46,-36,4,6,-2,134,36,-24,0,4,-2,-50,-13,8,0,-1,1},
  {-3,8,12,1,-1,0,8,-20,-30,-4,3,-1,1,0,-18,6,-4,1,-44,112,173,-18,-30,12,-31,77,112,0,-20,8,13,-31,-44,1,8,-3},
  {0,-1,1,12,8,-3,-1,3,-4,-30,-20,8,1,-4,6,-18,0,1,12,-30,-18,173,112,-44,8,-20,0,112,77,-31,-3,8,1,-44,-31,13},
  {0,0,0,-2,4,14,0,-1,1,6,-11,-38,0,1,-1,4,-6,-11,-2,6,4,-36,46,186,-2,4,0,-24,36,134,1,-1,0,8,-13,-50},
  {-3,-1,0,0,0,0,8,2,-1,0,0,0,2,1,-1,0,0,0,-38,-9,8,-1,-1,0,50,7,-9,1,2,-1,217,50,-38,2,8,-3},
  {1,-2,-2,0,0,0,-1,4,6,1,-1,0,0,0,4,-1,1,0,8,-24,-36,4,6,-2,-13,36,46,-6,-11,4,-50,134,186,-11,-38,14},
  {0,0,0,-2,-2,1,0,-1,1,6,4,-1,0,1,-1,4,0,0,-2,6,4,-36,-24,8,4,-11,-6,46,36,-13,14,-38,-11,186,134,-50},
  {0,0,0,0,-1,-3,0,0,0,-1,2,8,0,0,0,-1,1,2,0,-1,-1,8,-9,-38,-1,2,1,-9,7,50,-3,8,2,-38,50,217},
};

/* Farthest RGB pair over 36 texels; with skip_clear, fully transparent texels
 * (BC1 punch-through) do not pull the colour line. */
static inline void
astc_rgb_endpoints36(const uint8_t texels[144], int skip_clear, int *ai, int *bi)
{
   long best = -1;
   *ai = 0; *bi = 0;
   for (int i = 0; i < 36; i++) {
      if (skip_clear && texels[i * 4 + 3] == 0)
         continue;
      if (best < 0) { *ai = i; *bi = i; best = 0; }
      for (int j = i + 1; j < 36; j++) {
         if (skip_clear && texels[j * 4 + 3] == 0)
            continue;
         long d2 = 0;
         for (int c = 0; c < 3; c++) {
            long dc = (long)texels[i * 4 + c] - texels[j * 4 + c];
            d2 += dc * dc;
         }
         if (d2 > best) { best = d2; *ai = i; *bi = j; }
      }
   }
}

static inline void
astc_encode_rgb_6x6(const uint8_t texels[144], uint8_t out[16])
{
   memset(out, 0, 16);
   int ai, bi;
   astc_rgb_endpoints36(texels, 0, &ai, &bi);
   const uint8_t *pa = &texels[ai * 4], *pb = &texels[bi * 4];
   int sa = pa[0] + pa[1] + pa[2], sb = pb[0] + pb[1] + pb[2];
   const uint8_t *lo = (sb >= sa) ? pa : pb, *hi = (sb >= sa) ? pb : pa;
   int e0[3] = { lo[0], lo[1], lo[2] }, e1[3] = { hi[0], hi[1], hi[2] };

   astc_set_bits(out, 0, 11, 0x53);
   astc_set_bits(out, 13, 4, 8);
   astc_set_bits(out, 17 + 0, 8, e0[0]);  astc_set_bits(out, 17 + 8, 8, e1[0]);
   astc_set_bits(out, 17 + 16, 8, e0[1]); astc_set_bits(out, 17 + 24, 8, e1[1]);
   astc_set_bits(out, 17 + 32, 8, e0[2]); astc_set_bits(out, 17 + 40, 8, e1[2]);

   int dir[3]; long dlen2 = 0;
   for (int c = 0; c < 3; c++) { dir[c] = e1[c] - e0[c]; dlen2 += (long)dir[c] * dir[c]; }

   double ideal[36];
   for (int t = 0; t < 36; t++) {
      double w = 0;
      if (dlen2 != 0) {
         long dot = 0;
         for (int c = 0; c < 3; c++) dot += (long)(texels[t * 4 + c] - e0[c]) * dir[c];
         double f = (double)dot / (double)dlen2;
         if (f < 0) f = 0;
         if (f > 1) f = 1;
         w = f * 7.0;
      }
      ideal[t] = w;
   }
   for (int g = 0; g < 16; g++) {
      double a = 0;
      for (int t = 0; t < 36; t++) a += astc_pinv6[g][t] * ideal[t];
      int q = astc_round(a / 256.0);
      if (q < 0) q = 0;
      if (q > 7) q = 7;
      astc_set_bits(out, 128 - (g + 1) * 3, 3, astc_reverse_bits((uint32_t)q, 3));
   }
}

static inline void
astc_encode_rgba_6x6(const uint8_t texels[144], uint8_t out[16])
{
   memset(out, 0, 16);
   astc_set_bits(out, 0, 11, 0x442);
   astc_set_bits(out, 13, 4, 12);
   int amin = 255, amax = 0;
   for (int i = 0; i < 36; i++) { int a = texels[i * 4 + 3]; if (a < amin) amin = a; if (a > amax) amax = a; }
   int ai, bi;
   astc_rgb_endpoints36(texels, amin == 0 && amax > 0, &ai, &bi);
   const uint8_t *pa = &texels[ai * 4], *pb = &texels[bi * 4];
   int sa = pa[0] + pa[1] + pa[2], sb = pb[0] + pb[1] + pb[2];
   const uint8_t *lo = (sb >= sa) ? pa : pb, *hi = (sb >= sa) ? pb : pa;
   int e0[4] = { lo[0], lo[1], lo[2], amin }, e1[4] = { hi[0], hi[1], hi[2], amax };

   int q[8] = {
      astc_quantize_ce(e0[0]), astc_quantize_ce(e1[0]),
      astc_quantize_ce(e0[1]), astc_quantize_ce(e1[1]),
      astc_quantize_ce(e0[2]), astc_quantize_ce(e1[2]),
      astc_quantize_ce(e0[3]), astc_quantize_ce(e1[3]),
   };
   int s0 = astc_ce_unq[q[0]] + astc_ce_unq[q[2]] + astc_ce_unq[q[4]];
   int s1 = astc_ce_unq[q[1]] + astc_ce_unq[q[3]] + astc_ce_unq[q[5]];
   int swap = s1 < s0;
   if (swap)
      for (int k = 0; k < 8; k += 2) { int tmp = q[k]; q[k] = q[k + 1]; q[k + 1] = tmp; }
   astc_write_trit_endpoints(out, q);
   astc_set_bits(out, 62, 2, 3);

   int dir[3]; long dlen2 = 0;
   for (int c = 0; c < 3; c++) { dir[c] = e1[c] - e0[c]; dlen2 += (long)dir[c] * dir[c]; }
   int arange = amax - amin;

   double id0[36], id1[36];
   for (int t = 0; t < 36; t++) {
      double w0 = 0;
      if (dlen2 != 0) {
         long dot = 0;
         for (int c = 0; c < 3; c++) dot += (long)(texels[t * 4 + c] - e0[c]) * dir[c];
         double f = (double)dot / (double)dlen2;
         if (f < 0) f = 0;
         if (f > 1) f = 1;
         w0 = f * 3.0;
      }
      double w1 = 0;
      if (arange != 0) {
         double f = (double)(texels[t * 4 + 3] - amin) / (double)arange;
         if (f < 0) f = 0;
         if (f > 1) f = 1;
         w1 = f * 3.0;
      }
      if (swap) { w0 = 3.0 - w0; w1 = 3.0 - w1; }
      id0[t] = w0; id1[t] = w1;
   }
   for (int g = 0; g < 16; g++) {
      double a0 = 0, a1 = 0;
      for (int t = 0; t < 36; t++) { a0 += astc_pinv6[g][t] * id0[t]; a1 += astc_pinv6[g][t] * id1[t]; }
      int q0 = astc_round(a0 / 256.0), q1 = astc_round(a1 / 256.0);
      if (q0 < 0) q0 = 0;
      if (q0 > 3) q0 = 3;
      if (q1 < 0) q1 = 0;
      if (q1 > 3) q1 = 3;
      astc_set_bits(out, 128 - (2 * g + 1) * 2, 2, astc_reverse_bits((uint32_t)q0, 2));
      astc_set_bits(out, 128 - (2 * g + 2) * 2, 2, astc_reverse_bits((uint32_t)q1, 2));
   }
}

/* Encode a 6x6 RGBA8 block (36 texels, row-major) into a 16-byte ASTC block.
 * has_alpha blocks that turn out fully opaque take the finer RGB path. */
static inline void
astc_encode_block_6x6(const uint8_t texels[144], int has_alpha, uint8_t out[16])
{
   if (has_alpha) {
      for (int i = 0; i < 36; i++) {
         if (texels[i * 4 + 3] != 255) {
            astc_encode_rgba_6x6(texels, out);
            return;
         }
      }
   }
   astc_encode_rgb_6x6(texels, out);
}

/* ---- 4x4 encoder with a per-block mode search ----
 *
 * Tries the block configurations a hint lists (one or two partitions, one or two
 * weight planes, several weight ranges), fits endpoints per partition along the
 * principal axis with one least-squares refinement, and keeps the configuration
 * whose decoded texels are closest to the source. The colour endpoint range of
 * each configuration is the largest that fits the remaining bits, as the decoder
 * derives it. Block layout and quantization follow the Khronos Data Format
 * Specification, ASTC chapter (integer sequence encoding, unquantization tables,
 * partition hash). */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

/* A block whose squared error sums to this (RMS 1 per channel) ends the search. */
#define ASTC_GOOD 64

struct astc_qrange {
   uint16_t levels;
   uint8_t bits, trits, quints;
};

static const struct astc_qrange astc_qr[21] = {
   { 2, 1, 0, 0 },   { 3, 0, 1, 0 },   { 4, 2, 0, 0 },   { 5, 0, 0, 1 },
   { 6, 1, 1, 0 },   { 8, 3, 0, 0 },   { 10, 1, 0, 1 },  { 12, 2, 1, 0 },
   { 16, 4, 0, 0 },  { 20, 2, 0, 1 },  { 24, 3, 1, 0 },  { 32, 5, 0, 0 },
   { 40, 3, 0, 1 },  { 48, 4, 1, 0 },  { 64, 6, 0, 0 },  { 80, 4, 0, 1 },
   { 96, 5, 1, 0 },  { 128, 7, 0, 0 }, { 160, 5, 0, 1 }, { 192, 6, 1, 0 },
   { 256, 8, 0, 0 },
};

static struct {
   uint8_t trit_enc[243];
   uint8_t quint_enc[125];
   uint8_t wunq[12][32];
   uint8_t wsorted[12][32];
   uint8_t wnear[12][65];
   uint8_t cunq[21][256];
   uint8_t cquant[21][256];
   uint16_t pmask2[1024];
} astc_t;

static pthread_once_t astc_t_once = PTHREAD_ONCE_INIT;

static inline int
astc_ise_size(int count, int q)
{
   const struct astc_qrange *r = &astc_qr[q];
   return count * r->bits + (r->trits ? (count * 8 + 4) / 5 : 0) +
          (r->quints ? (count * 7 + 2) / 3 : 0);
}

static void
astc_trit_decode(int T, int t[5])
{
   int C;
   if (((T >> 2) & 7) == 7) {
      C = (((T >> 5) & 7) << 2) | (T & 3);
      t[4] = 2;
      t[3] = 2;
   } else {
      C = T & 0x1f;
      if (((T >> 5) & 3) == 3) {
         t[4] = 2;
         t[3] = (T >> 7) & 1;
      } else {
         t[4] = (T >> 7) & 1;
         t[3] = (T >> 5) & 3;
      }
   }
   if ((C & 3) == 3) {
      t[2] = 2;
      t[1] = (C >> 4) & 1;
      t[0] = (((C >> 3) & 1) << 1) | (((C >> 2) & 1) & ~(C >> 3) & 1);
   } else if (((C >> 2) & 3) == 3) {
      t[2] = 2;
      t[1] = 2;
      t[0] = C & 3;
   } else {
      t[2] = (C >> 4) & 1;
      t[1] = (C >> 2) & 3;
      t[0] = (((C >> 1) & 1) << 1) | ((C & 1) & ~(C >> 1) & 1);
   }
}

static void
astc_quint_decode(int Q, int q[3])
{
   if (((Q >> 1) & 3) == 3 && ((Q >> 5) & 3) == 0) {
      int b0 = Q & 1;
      q[2] = (b0 << 2) | ((((Q >> 4) & 1) & ~b0 & 1) << 1) | (((Q >> 3) & 1) & ~b0 & 1);
      q[1] = 4;
      q[0] = 4;
      return;
   }
   int C;
   if (((Q >> 1) & 3) == 3) {
      q[2] = 4;
      C = (((Q >> 3) & 3) << 3) | ((~(Q >> 5) & 3) << 1) | (Q & 1);
   } else {
      q[2] = (Q >> 5) & 3;
      C = Q & 0x1f;
   }
   if ((C & 7) == 5) {
      q[1] = 4;
      q[0] = (C >> 3) & 3;
   } else {
      q[1] = (C >> 3) & 3;
      q[0] = C & 7;
   }
}

static int
astc_unq_weight(int q, int v)
{
   const struct astc_qrange *r = &astc_qr[q];
   int b = r->bits, out;
   if (!r->trits && !r->quints) {
      out = 0;
      for (int sh = 6 - b; sh > -b; sh -= b)
         out |= sh >= 0 ? v << sh : v >> -sh;
   } else if (b == 0) {
      static const uint8_t t3[3] = { 0, 32, 63 }, q5[5] = { 0, 16, 32, 47, 63 };
      out = r->trits ? t3[v] : q5[v];
   } else {
      int m = v & ((1 << b) - 1), D = v >> b;
      int A = (m & 1) ? 0x7f : 0, B = 0, C;
      int bb = (m >> 1) & 1, cc = (m >> 2) & 1;
      if (r->trits)
         C = b == 1 ? 50 : b == 2 ? 23 : 11;
      else
         C = b == 1 ? 28 : 13;
      if (r->trits && b == 2)
         B = bb ? 0x45 : 0;
      else if (r->trits && b == 3)
         B = (cc << 6) | (bb << 5) | (cc << 1) | bb;
      else if (r->quints && b == 2)
         B = bb ? 0x42 : 0;
      int T = (D * C + B) ^ A;
      out = (A & 0x20) | (T >> 2);
   }
   return out > 32 ? out + 1 : out;
}

static int
astc_unq_color(int q, int v)
{
   const struct astc_qrange *r = &astc_qr[q];
   int b = r->bits;
   if (!r->trits && !r->quints) {
      int out = 0;
      for (int sh = 8 - b; sh > -b; sh -= b)
         out |= sh >= 0 ? v << sh : v >> -sh;
      return out;
   }
   int m = v & ((1 << b) - 1), D = v >> b;
   int A = (m & 1) ? 0x1ff : 0, B = 0, C;
   int b1 = (m >> 1) & 1, c = (m >> 2) & 1, d = (m >> 3) & 1, e = (m >> 4) & 1, f = (m >> 5) & 1;
   if (r->trits) {
      switch (b) {
      case 1: C = 204; break;
      case 2: C = 93; B = (b1 << 8) | (b1 << 4) | (b1 << 2) | (b1 << 1); break;
      case 3: C = 44; B = (c << 8) | (b1 << 7) | (c << 3) | (b1 << 2) | (c << 1) | b1; break;
      case 4: C = 22; B = (d << 8) | (c << 7) | (b1 << 6) | (d << 2) | (c << 1) | b1; break;
      case 5: C = 11; B = (e << 8) | (d << 7) | (c << 6) | (b1 << 5) | (e << 1) | d; break;
      default: C = 5; B = (f << 8) | (e << 7) | (d << 6) | (c << 5) | (b1 << 4) | f; break;
      }
   } else {
      switch (b) {
      case 1: C = 113; break;
      case 2: C = 54; B = (b1 << 8) | (b1 << 3) | (b1 << 2); break;
      case 3: C = 26; B = (c << 8) | (b1 << 7) | (c << 2) | (b1 << 1) | c; break;
      case 4: C = 13; B = (d << 8) | (c << 7) | (b1 << 6) | (d << 1) | c; break;
      default: C = 6; B = (e << 8) | (d << 7) | (c << 6) | (b1 << 5) | e; break;
      }
   }
   int T = (D * C + B) ^ A;
   return (A & 0x80) | (T >> 2);
}

static int
astc_partition_of(int seed, int x, int y, int count)
{
   x <<= 1;
   y <<= 1;
   seed += (count - 1) * 1024;
   uint32_t r = (uint32_t)seed;
   r ^= r >> 15;
   r *= 0xEEDE0891u;
   r ^= r >> 5;
   r += r << 16;
   r ^= r >> 7;
   r ^= r >> 3;
   r ^= r << 6;
   r ^= r >> 17;
   int s[12];
   static const int shifts[11] = { 0, 4, 8, 12, 16, 20, 24, 28, 18, 22, 26 };
   for (int i = 0; i < 11; i++)
      s[i] = (r >> shifts[i]) & 0xf;
   s[11] = ((r >> 30) | (r << 2)) & 0xf;
   for (int i = 0; i < 12; i++)
      s[i] *= s[i];
   int sh1, sh2;
   if (seed & 1) {
      sh1 = (seed & 2) ? 4 : 5;
      sh2 = count == 3 ? 6 : 5;
   } else {
      sh1 = count == 3 ? 6 : 5;
      sh2 = (seed & 2) ? 4 : 5;
   }
   int sh3 = (seed & 0x10) ? sh1 : sh2;
   for (int i = 0; i < 8; i++)
      s[i] >>= (i & 1) ? sh2 : sh1;
   for (int i = 8; i < 12; i++)
      s[i] >>= sh3;
   int a = (s[0] * x + s[1] * y + (r >> 14)) & 0x3f;
   int b = (s[2] * x + s[3] * y + (r >> 10)) & 0x3f;
   int c = count >= 3 ? (int)((s[4] * x + s[5] * y + (r >> 6)) & 0x3f) : 0;
   if (a >= b && a >= c)
      return 0;
   return b >= c ? 1 : 2;
}

static void
astc_t_init(void)
{
   uint8_t seen[243] = { 0 };
   for (int T = 0; T < 256; T++) {
      int t[5];
      astc_trit_decode(T, t);
      int idx = t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4];
      if (!seen[idx]) {
         seen[idx] = 1;
         astc_t.trit_enc[idx] = T;
      }
   }
   memset(seen, 0, sizeof(seen));
   for (int Q = 0; Q < 128; Q++) {
      int q[3];
      astc_quint_decode(Q, q);
      int idx = q[0] + 5 * q[1] + 25 * q[2];
      if (!seen[idx]) {
         seen[idx] = 1;
         astc_t.quint_enc[idx] = Q;
      }
   }
   for (int q = 0; q < 12; q++) {
      int n = astc_qr[q].levels;
      for (int v = 0; v < n; v++)
         astc_t.wunq[q][v] = astc_unq_weight(q, v);
      for (int v = 0; v < n; v++)
         astc_t.wsorted[q][v] = v;
      for (int i = 1; i < n; i++)
         for (int j = i; j > 0 &&
              astc_t.wunq[q][astc_t.wsorted[q][j]] < astc_t.wunq[q][astc_t.wsorted[q][j - 1]]; j--) {
            uint8_t tmp = astc_t.wsorted[q][j];
            astc_t.wsorted[q][j] = astc_t.wsorted[q][j - 1];
            astc_t.wsorted[q][j - 1] = tmp;
         }
      for (int w = 0; w <= 64; w++) {
         int best = 0, bd = 1 << 30;
         for (int k = 0; k < n; k++) {
            int d = abs((int)astc_t.wunq[q][astc_t.wsorted[q][k]] - w);
            if (d < bd) {
               bd = d;
               best = k;
            }
         }
         astc_t.wnear[q][w] = best;
      }
   }
   for (int seed = 0; seed < 1024; seed++) {
      uint16_t m = 0;
      for (int i = 0; i < 16; i++)
         m |= astc_partition_of(seed, i & 3, i >> 2, 2) << i;
      astc_t.pmask2[seed] = m;
   }
   for (int q = 4; q < 21; q++) {
      int n = astc_qr[q].levels;
      for (int v = 0; v < n; v++)
         astc_t.cunq[q][v] = astc_unq_color(q, v);
      for (int x = 0; x < 256; x++) {
         int best = 0, bd = 1 << 30;
         for (int v = 0; v < n; v++) {
            int d = abs((int)astc_t.cunq[q][v] - x);
            if (d < bd) {
               bd = d;
               best = v;
            }
         }
         astc_t.cquant[q][x] = best;
      }
   }
}

/* Writes an integer sequence of `count` values from bit `pos`; reverse=1 stores it
 * bit-reversed from bit 127 down (the weight layout). */
static void
astc_ise_write(uint8_t blk[16], int pos, const uint8_t *vals, int count, int q, int reverse)
{
   const struct astc_qrange *r = &astc_qr[q];
   uint8_t tmp[40];
   int p = 0, b = r->bits;
   memset(tmp, 0, sizeof(tmp));
   if (r->trits) {
      static const int tb[5] = { 2, 2, 1, 2, 1 };
      for (int g = 0; g < count; g += 5) {
         int t[5] = { 0 }, m[5] = { 0 };
         for (int j = 0; j < 5 && g + j < count; j++) {
            t[j] = vals[g + j] >> b;
            m[j] = vals[g + j] & ((1 << b) - 1);
         }
         int T = astc_t.trit_enc[t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4]];
         int ts = 0;
         for (int j = 0; j < 5; j++) {
            astc_set_bits(tmp, p, b, m[j]);
            p += b;
            astc_set_bits(tmp, p, tb[j], (T >> ts) & ((1 << tb[j]) - 1));
            p += tb[j];
            ts += tb[j];
         }
      }
   } else if (r->quints) {
      static const int qb[3] = { 3, 2, 2 };
      for (int g = 0; g < count; g += 3) {
         int qv[3] = { 0 }, m[3] = { 0 };
         for (int j = 0; j < 3 && g + j < count; j++) {
            qv[j] = vals[g + j] >> b;
            m[j] = vals[g + j] & ((1 << b) - 1);
         }
         int Q = astc_t.quint_enc[qv[0] + 5 * qv[1] + 25 * qv[2]];
         int qs = 0;
         for (int j = 0; j < 3; j++) {
            astc_set_bits(tmp, p, b, m[j]);
            p += b;
            astc_set_bits(tmp, p, qb[j], (Q >> qs) & ((1 << qb[j]) - 1));
            p += qb[j];
            qs += qb[j];
         }
      }
   } else {
      for (int j = 0; j < count; j++) {
         astc_set_bits(tmp, p, b, vals[j]);
         p += b;
      }
   }
   int len = astc_ise_size(count, q);
   for (int i = 0; i < len; i++)
      if ((tmp[i >> 3] >> (i & 7)) & 1)
         astc_set_bits(blk, reverse ? 127 - i : pos + i, 1, 1);
}


/* BC7 partition tables: 2 subsets as a mask (bit i = texel i in subset 1), 3
 * subsets as 2 bits per texel. */
static const uint16_t astc_bc7_part2[64] = {
   0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
   0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
   0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
   0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
   0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
   0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
   0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
   0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
};

static const uint32_t astc_bc7_part3[64] = {
   0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050,
   0x5555a0a0, 0x5a5a5050, 0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090,
   0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250, 0xa5945040, 0x0a425054,
   0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
   0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414,
   0x50a4a450, 0x6a5a0200, 0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424,
   0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50, 0x500aa550, 0xaaaa4444,
   0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
   0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580,
   0xaa141414, 0x96960000, 0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
   0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};

/* ASTC 4x4 three-partition seed closest to each BC7 three-subset partition. */
static const uint16_t astc_bc7_seed3[64] = {
   457, 460, 260, 0, 260, 738, 8, 460, 74, 32, 156, 183, 15, 745, 1012, 90,
   502, 488, 247, 738, 0, 457, 181, 260, 90, 475, 704, 637, 108, 104, 254, 260,
   276, 738, 89, 335, 902, 36, 516, 472, 262, 104, 49, 760, 738, 459, 70, 907,
   368, 182, 730, 483, 26, 562, 368, 637, 104, 254, 760, 388, 163, 837, 590, 488,
};

struct astc_hint {
   int srgb;
   int alpha;
   int dual_ccs;
   int seed3;
   int nmasks, clear;
   uint16_t masks[4];
   uint16_t single_wq, dual_wq, part_wq;
};

/* Up to `n` two-partition seeds whose texel split is closest to `mask` (bit i =
 * texel i), all at the smallest distance found. */
static int
astc_seeds_for_mask(uint16_t mask, int *seeds, int n)
{
   int dist[8], found = 0;
   for (int seed = 0; seed < 1024; seed++) {
      uint16_t m = astc_t.pmask2[seed];
      if (!m || m == 0xffff)
         continue;
      int d = __builtin_popcount((m ^ mask) & 0xffff);
      d = d < 16 - d ? d : 16 - d;
      int k = found < n ? found++ : n;
      while (k > 0 && dist[k - 1] > d) {
         if (k < n) {
            dist[k] = dist[k - 1];
            seeds[k] = seeds[k - 1];
         }
         k--;
      }
      if (k < n) {
         dist[k] = d;
         seeds[k] = seed;
      }
   }
   while (found > 1 && dist[found - 1] > dist[0])
      found--;
   return found;
}

static inline int
astc_dec8(int u0, int u1, int w, int srgb)
{
   int e0 = srgb ? (u0 << 8) | 0x80 : u0 * 257;
   int e1 = srgb ? (u1 << 8) | 0x80 : u1 * 257;
   int c = (e0 * (64 - w) + e1 * w + 32) >> 6;
   return srgb ? c >> 8 : (c * 255 + 32767) / 65535;
}

struct astc_trial {
   int parts, seed, cem, dual, ccs, wq, cq, nch, clear, punch, srgb;
   uint8_t label[16];
   int u0[3][4], u1[3][4];
   uint8_t q0[3][4], q1[3][4];
   uint8_t w[2][16];
   uint8_t dec[3][4][32];
};

static void
astc_quantize_endpoints(struct astc_trial *t, int p, const float e0[4], const float e1[4])
{
   for (int c = 0; c < t->nch; c++) {
      int a = (int)(e0[c] + 0.5f), b = (int)(e1[c] + 0.5f);
      a = a < 0 ? 0 : a > 255 ? 255 : a;
      b = b < 0 ? 0 : b > 255 ? 255 : b;
      t->q0[p][c] = astc_t.cquant[t->cq][a];
      t->q1[p][c] = astc_t.cquant[t->cq][b];
      t->u0[p][c] = astc_t.cunq[t->cq][t->q0[p][c]];
      t->u1[p][c] = astc_t.cunq[t->cq][t->q1[p][c]];
   }
   if (t->u1[p][0] + t->u1[p][1] + t->u1[p][2] < t->u0[p][0] + t->u0[p][1] + t->u0[p][2]) {
      for (int c = 0; c < t->nch; c++) {
         int x = t->u0[p][c]; t->u0[p][c] = t->u1[p][c]; t->u1[p][c] = x;
         uint8_t y = t->q0[p][c]; t->q0[p][c] = t->q1[p][c]; t->q1[p][c] = y;
      }
   }
}

/* Decoded 8-bit value of every weight value per partition and channel. */
static void
astc_build_dec(struct astc_trial *t)
{
   int n = astc_qr[t->wq].levels;
   for (int p = 0; p < t->parts; p++) {
      if (p == t->clear)
         continue;
      for (int c = 0; c < t->nch; c++)
         for (int v = 0; v < n; v++)
            t->dec[p][c][v] = astc_dec8(t->u0[p][c], t->u1[p][c], astc_t.wunq[t->wq][v],
                                        t->srgb && c < 3);
   }
}

/* Best weight per texel for the current endpoints; returns the block error. */
static long
astc_fit_weights(struct astc_trial *t, const uint8_t tex[64])
{
   long sse = 0;
   int n = astc_qr[t->wq].levels;
   astc_build_dec(t);
   for (int i = 0; i < 16; i++) {
      int p = t->label[i];
      const int *u0 = t->u0[p], *u1 = t->u1[p];
      if (p == t->clear) {
         t->w[0][i] = 0;
         for (int c = 0; c < 4; c++)
            sse += tex[i * 4 + c] * tex[i * 4 + c];
         continue;
      }
      for (int plane = 0; plane <= t->dual; plane++) {
         float dot = 0, len = 0;
         for (int c = 0; c < t->nch; c++) {
            if (t->dual && ((c == t->ccs) != plane))
               continue;
            float d = (float)(u1[c] - u0[c]);
            dot += ((float)tex[i * 4 + c] - u0[c]) * d;
            len += d * d;
         }
         float f = len > 0 ? dot / len : 0;
         f = f < 0 ? 0 : f > 1 ? 1 : f;
         int k0 = astc_t.wnear[t->wq][(int)(f * 64 + 0.5f)];
         long best = -1;
         int bv = 0;
         for (int k = k0 - 1; k <= k0 + 1; k++) {
            if (k < 0 || k >= n)
               continue;
            int v = astc_t.wsorted[t->wq][k];
            long e = 0;
            for (int c = 0; c < t->nch; c++) {
               if (t->dual && ((c == t->ccs) != plane))
                  continue;
               int d = t->dec[p][c][v] - tex[i * 4 + c];
               e += d * d;
            }
            if (best < 0 || e < best) {
               best = e;
               bv = v;
            }
         }
         t->w[plane][i] = bv;
         sse += best;
      }
      if (t->nch == 3) {
         int d = 255 - tex[i * 4 + 3];
         sse += d * d;
      }
   }
   return sse;
}

static void
astc_fit_line(const struct astc_trial *t, const uint8_t tex[64], int p, float e0[4], float e1[4])
{
   float mean[4] = { 0 }, cov[4][4] = { { 0 } };
   int n = 0, line[4], nl = 0;
   for (int c = 0; c < t->nch; c++)
      if (!(t->dual && c == t->ccs))
         line[nl++] = c;
   float lo[4] = { 255, 255, 255, 255 }, hi[4] = { 0 };
   int skip_clear = 0;
   if (t->punch && t->dual && t->ccs == 3) {
      int clear = 0, solid = 0;
      for (int i = 0; i < 16; i++)
         if (t->label[i] == p) {
            clear += tex[i * 4 + 3] == 0;
            solid += tex[i * 4 + 3] != 0;
         }
      skip_clear = clear && solid;
   }
   for (int i = 0; i < 16; i++) {
      if (t->label[i] != p || (skip_clear && tex[i * 4 + 3] == 0))
         continue;
      n++;
      for (int c = 0; c < t->nch; c++) {
         float v = tex[i * 4 + c];
         mean[c] += v;
         lo[c] = v < lo[c] ? v : lo[c];
         hi[c] = v > hi[c] ? v : hi[c];
      }
   }
   if (!n) {
      for (int c = 0; c < 4; c++)
         e0[c] = e1[c] = 0;
      return;
   }
   for (int c = 0; c < t->nch; c++)
      mean[c] /= n;
   for (int i = 0; i < 16; i++) {
      if (t->label[i] != p || (skip_clear && tex[i * 4 + 3] == 0))
         continue;
      for (int a = 0; a < nl; a++)
         for (int b = 0; b < nl; b++)
            cov[a][b] += (tex[i * 4 + line[a]] - mean[line[a]]) *
                         (tex[i * 4 + line[b]] - mean[line[b]]);
   }
   float axis[4];
   for (int a = 0; a < nl; a++)
      axis[a] = hi[line[a]] - lo[line[a]];
   for (int it = 0; it < 4; it++) {
      float nx[4] = { 0 }, m = 0;
      for (int a = 0; a < nl; a++)
         for (int b = 0; b < nl; b++)
            nx[a] += cov[a][b] * axis[b];
      for (int a = 0; a < nl; a++)
         m = fabsf(nx[a]) > m ? fabsf(nx[a]) : m;
      if (m <= 0)
         break;
      for (int a = 0; a < nl; a++)
         axis[a] = nx[a] / m;
   }
   float len = 0;
   for (int a = 0; a < nl; a++)
      len += axis[a] * axis[a];
   float tmin = 0, tmax = 0;
   if (len > 0) {
      tmin = 1e30f;
      tmax = -1e30f;
      for (int i = 0; i < 16; i++) {
         if (t->label[i] != p || (skip_clear && tex[i * 4 + 3] == 0))
            continue;
         float d = 0;
         for (int a = 0; a < nl; a++)
            d += (tex[i * 4 + line[a]] - mean[line[a]]) * axis[a];
         d /= len;
         tmin = d < tmin ? d : tmin;
         tmax = d > tmax ? d : tmax;
      }
   }
   for (int a = 0; a < nl; a++) {
      e0[line[a]] = mean[line[a]] + tmin * axis[a];
      e1[line[a]] = mean[line[a]] + tmax * axis[a];
   }
   if (t->dual) {
      e0[t->ccs] = skip_clear ? 0 : lo[t->ccs];
      e1[t->ccs] = hi[t->ccs];
   }
}

/* Least-squares endpoints for the weights chosen so far. */
static void
astc_refit(const struct astc_trial *t, const uint8_t tex[64], int p, float e0[4], float e1[4])
{
   for (int c = 0; c < t->nch; c++) {
      int plane = t->dual && c == t->ccs;
      float aa = 0, ab = 0, bb = 0, ax = 0, bx = 0;
      for (int i = 0; i < 16; i++) {
         if (t->label[i] != p)
            continue;
         float w = astc_t.wunq[t->wq][t->w[plane][i]] / 64.0f, x = tex[i * 4 + c];
         aa += (1 - w) * (1 - w);
         ab += (1 - w) * w;
         bb += w * w;
         ax += (1 - w) * x;
         bx += w * x;
      }
      float det = aa * bb - ab * ab;
      if (fabsf(det) < 1e-6f) {
         e0[c] = t->u0[p][c];
         e1[c] = t->u1[p][c];
         continue;
      }
      e0[c] = (ax * bb - bx * ab) / det;
      e1[c] = (bx * aa - ax * ab) / det;
   }
}

static long
astc_run_trial(struct astc_trial *t, const uint8_t tex[64], int srgb)
{
   t->srgb = srgb;
   int header = t->parts == 1 ? 17 : 29;
   int nw = t->dual ? 32 : 16;
   int wbits = astc_ise_size(nw, t->wq);
   if (wbits < 24 || wbits > 96)
      return -1;
   for (int i = 0; i < 16; i++)
      t->label[i] = t->parts == 1 ? 0 : astc_partition_of(t->seed, i & 3, i >> 2, t->parts);
   if (t->clear == -2) {
      int clear[3] = { 0 }, all[3] = { 0 };
      for (int i = 0; i < 16; i++) {
         all[t->label[i]]++;
         clear[t->label[i]] += tex[i * 4 + 3] == 0;
      }
      t->clear = clear[0] * 2 > all[0] ? 0 : 1;
      if (t->parts != 2 || t->cem != 8 || clear[t->clear] * 2 <= all[t->clear])
         return -1;
   }
   int avail = 128 - header - wbits - (t->dual ? 2 : 0) - (t->clear >= 0 ? 2 : 0);
   int nvals = t->parts * (t->cem == 12 ? 8 : 6) - (t->clear >= 0 ? 2 : 0);
   t->cq = -1;
   for (int q = 20; q >= 4; q--)
      if (astc_ise_size(nvals, q) <= avail) {
         t->cq = q;
         break;
      }
   if (t->cq < 0)
      return -1;
   t->nch = t->cem == 12 ? 4 : 3;

   for (int p = 0; p < t->parts; p++) {
      float e0[4], e1[4];
      if (p == t->clear)
         continue;
      astc_fit_line(t, tex, p, e0, e1);
      astc_quantize_endpoints(t, p, e0, e1);
   }
   long err = astc_fit_weights(t, tex);
   if (err <= ASTC_GOOD)
      return err;

   struct astc_trial r = *t;
   for (int p = 0; p < r.parts; p++) {
      float e0[4], e1[4];
      if (p == r.clear)
         continue;
      astc_refit(&r, tex, p, e0, e1);
      astc_quantize_endpoints(&r, p, e0, e1);
   }
   long rerr = astc_fit_weights(&r, tex);
   if (rerr < err) {
      *t = r;
      err = rerr;
   }
   return err;
}

static void
astc_write_trial(const struct astc_trial *t, uint8_t out[16])
{
   memset(out, 0, 16);
   int R = t->wq < 6 ? t->wq + 2 : t->wq - 4, H = t->wq >= 6;
   int mode = ((R >> 1) & 3) | ((R & 1) << 4) | (2 << 5) | (H << 9) | (t->dual << 10);
   astc_set_bits(out, 0, 11, mode);
   astc_set_bits(out, 11, 2, t->parts - 1);
   int pos;
   if (t->parts == 1) {
      astc_set_bits(out, 13, 4, t->cem);
      pos = 17;
   } else if (t->clear >= 0) {
      astc_set_bits(out, 13, 10, t->seed);
      astc_set_bits(out, 23, 2, 2);
      astc_set_bits(out, 25 + t->clear, 1, 0);
      astc_set_bits(out, 25 + !t->clear, 1, 1);
      pos = 29;
   } else {
      astc_set_bits(out, 13, 10, t->seed);
      astc_set_bits(out, 23, 2, 0);
      astc_set_bits(out, 25, 4, t->cem);
      pos = 29;
   }
   uint8_t vals[24];
   int n = 0;
   for (int p = 0; p < t->parts; p++) {
      if (p == t->clear) {
         memset(vals + n, 0, 4);
         n += 4;
         continue;
      }
      for (int c = 0; c < t->nch; c++) {
         vals[n++] = t->q0[p][c];
         vals[n++] = t->q1[p][c];
      }
   }
   astc_ise_write(out, pos, vals, n, t->cq, 0);
   int nw = t->dual ? 32 : 16;
   uint8_t wv[32];
   for (int i = 0; i < 16; i++) {
      if (t->dual) {
         wv[i * 2] = t->w[0][i];
         wv[i * 2 + 1] = t->w[1][i];
      } else {
         wv[i] = t->w[0][i];
      }
   }
   int wbits = astc_ise_size(nw, t->wq);
   if (t->dual)
      astc_set_bits(out, 128 - wbits - 2, 2, t->ccs);
   astc_ise_write(out, 0, wv, nw, t->wq, 1);
}

static void
astc_write_void_extent(const uint8_t rgba[4], uint8_t out[16])
{
   static const uint8_t head[8] = { 0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
   memcpy(out, head, 8);
   for (int c = 0; c < 4; c++) {
      out[8 + 2 * c] = rgba[c];
      out[9 + 2 * c] = rgba[c];
   }
}

static void
astc_encode_4x4_search(const uint8_t tex[64], const struct astc_hint *h, uint8_t out[16])
{
   pthread_once(&astc_t_once, astc_t_init);

   uint8_t px[64];
   memcpy(px, tex, 64);
   int opaque = 1, flat = 1;
   for (int i = 0; i < 16; i++) {
      if (!h->alpha)
         px[i * 4 + 3] = 255;
      opaque &= px[i * 4 + 3] == 255;
      flat &= !memcmp(px + i * 4, px, 4);
   }
   if (flat) {
      astc_write_void_extent(px, out);
      return;
   }

   struct astc_trial best, t;
   long best_err = LONG_MAX;
   int cem = opaque ? 8 : 12;

#define ASTC_TRY(P, SEED, CEM, DUAL, CCS, WQ, CLEAR)                              \
   do {                                                                         \
      memset(&t, 0, sizeof(t));                                                 \
      t.parts = (P); t.seed = (SEED); t.cem = (CEM); t.dual = (DUAL);           \
      t.ccs = (CCS); t.wq = (WQ); t.clear = (CLEAR); t.punch = h->clear;        \
      long e = astc_run_trial(&t, px, h->srgb);                                 \
      if (e >= 0 && e < best_err) {                                             \
         best = t;                                                              \
         best_err = e;                                                          \
      }                                                                         \
   } while (0)

   for (int wq = 0; wq < 12 && best_err > ASTC_GOOD; wq++)
      if ((h->single_wq >> wq) & 1)
         ASTC_TRY(1, 0, cem, 0, 0, wq, -1);
   if (h->dual_ccs >= 0 && (h->dual_ccs < 3 || !opaque)) {
      int dcem = h->dual_ccs == 3 ? 12 : cem;
      for (int wq = 0; wq < 12 && best_err > ASTC_GOOD; wq++)
         if ((h->dual_wq >> wq) & 1)
            ASTC_TRY(1, 0, dcem, 1, h->dual_ccs, wq, -1);
   }
   for (int m = 0; m < h->nmasks && best_err > ASTC_GOOD; m++) {
      int seeds[4], n = astc_seeds_for_mask(h->masks[m], seeds, h->clear || h->nmasks == 1 ? 4 : 2);
      for (int k = 0; k < n; k++)
         for (int wq = 0; wq < 12; wq++) {
            if (!((h->part_wq >> wq) & 1))
               continue;
            ASTC_TRY(2, seeds[k], cem, 0, 0, wq, -1);
            if (h->clear)
               ASTC_TRY(2, seeds[k], 8, 0, 0, wq, -2);
         }
   }
   for (int wq = 0; wq < 12 && h->seed3 >= 0 && best_err > ASTC_GOOD; wq++)
      if ((h->part_wq >> wq) & 1)
         ASTC_TRY(3, h->seed3, cem, 0, 0, wq, -1);
#undef ASTC_TRY

   if (best_err == LONG_MAX) {
      astc_encode_block_4x4(px, !opaque, out);
      return;
   }
   astc_write_trial(&best, out);
}

#endif /* WRAPPER_ASTC_H */
