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

/* RG dual-plane block (BC5 sources): CEM 12, 2-bit weights per plane, green on
 * plane 1 so red and green keep independent weights instead of being projected
 * onto one RGB line. Blue endpoints are 0 and alpha endpoints 255, so those two
 * components stay constant whatever the weights are. */
static inline void
astc_encode_rg_4x4(const uint8_t texels[64], uint8_t out[16])
{
   memset(out, 0, 16);
   astc_set_bits(out, 0, 11, 0x442);
   astc_set_bits(out, 13, 4, 12);

   int rmin = 255, rmax = 0, gmin = 255, gmax = 0;
   for (int i = 0; i < 16; i++) {
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
   astc_set_bits(out, 62, 2, 1);   /* CCS = green component */

   int r0 = astc_ce_unq[q[0]], r1 = astc_ce_unq[q[1]];
   int g0 = astc_ce_unq[q[2]], g1 = astc_ce_unq[q[3]];
   int rrange = r1 - r0, grange = g1 - g0;

   for (int px = 0; px < 16; px++) {
      int q0 = 0, q1 = 0;
      if (rrange > 0) {
         int v = texels[px * 4] - r0;
         q0 = (v <= 0) ? 0 : (v >= rrange ? 3 : (v * 3 + rrange / 2) / rrange);
      }
      if (grange > 0) {
         int v = texels[px * 4 + 1] - g0;
         q1 = (v <= 0) ? 0 : (v >= grange ? 3 : (v * 3 + grange / 2) / grange);
      }
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

#endif /* WRAPPER_ASTC_H */
