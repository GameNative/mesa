#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "wrapper_spirv_lower.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define SPV_MAGIC 0x07230203u
#define SPV_1_3 0x00010300u

enum {
   OP_EXTENSION = 10,
   OP_EXT_INST_IMPORT = 11,
   OP_MEMORY_MODEL = 14,
   OP_ENTRY_POINT = 15,
   OP_EXECUTION_MODE = 16,
   OP_CAPABILITY = 17,
   OP_TYPE_ARRAY = 28,
   OP_TYPE_RUNTIME_ARRAY = 29,
   OP_TYPE_STRUCT = 30,
   OP_TYPE_POINTER = 32,
   OP_CONSTANT = 43,
   OP_VARIABLE = 59,
   OP_LOAD = 61,
   OP_DECORATE = 71,
   OP_MEMBER_DECORATE = 72,
   OP_COPY_OBJECT = 83,
   OP_COMPOSITE_CONSTRUCT = 80,
   OP_COMPOSITE_EXTRACT = 81,
   OP_SELECT = 169,
   OP_KILL = 252,
   OP_COPY_LOGICAL = 400,
   OP_PTR_EQUAL = 401,
   OP_PTR_NOT_EQUAL = 402,
   OP_PTR_DIFF = 403,
   OP_TERMINATE_INVOCATION = 4416,
};

enum {
   CAP_SHADER_LAYER = 69,
   CAP_SHADER_VIEWPORT_INDEX = 70,
   CAP_STORAGE_BUFFER_8BIT = 4448,
   CAP_UNIFORM_STORAGE_BUFFER_8BIT = 4449,
   CAP_STORAGE_PUSH_CONSTANT_8 = 4450,
   CAP_DENORM_PRESERVE = 4464,
   CAP_ROUNDING_MODE_RTZ = 4468,
   CAP_SHADER_VIEWPORT_INDEX_LAYER = 5254,
   CAP_SHADER_NON_UNIFORM = 5301,
   CAP_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM = 5312,
   CAP_VULKAN_MEMORY_MODEL = 5345,
   CAP_VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 5346,
   CAP_PHYSICAL_STORAGE_BUFFER = 5347,
   CAP_DEMOTE_TO_HELPER = 5379,
   CAP_DOT_PRODUCT_INPUT_ALL = 6016,
   CAP_DOT_PRODUCT = 6019,
};

enum {
   EXT_VIEWPORT_LAYER,
   EXT_8BIT,
   EXT_FLOAT_CONTROLS,
   EXT_DESCRIPTOR_INDEXING,
   EXT_VMM,
   EXT_PHYSICAL_STORAGE,
   EXT_DEMOTE,
   EXT_DOT_PRODUCT,
   EXT_NON_SEMANTIC,
   EXT_COUNT,
};

static const char *const ext_names[EXT_COUNT] = {
   [EXT_VIEWPORT_LAYER] = "SPV_EXT_shader_viewport_index_layer",
   [EXT_8BIT] = "SPV_KHR_8bit_storage",
   [EXT_FLOAT_CONTROLS] = "SPV_KHR_float_controls",
   [EXT_DESCRIPTOR_INDEXING] = "SPV_EXT_descriptor_indexing",
   [EXT_VMM] = "SPV_KHR_vulkan_memory_model",
   [EXT_PHYSICAL_STORAGE] = "SPV_KHR_physical_storage_buffer",
   [EXT_DEMOTE] = "SPV_EXT_demote_to_helper_invocation",
   [EXT_DOT_PRODUCT] = "SPV_KHR_integer_dot_product",
   [EXT_NON_SEMANTIC] = "SPV_KHR_non_semantic_info",
};

static int
cap_extension(uint32_t cap)
{
   if (cap == CAP_SHADER_LAYER || cap == CAP_SHADER_VIEWPORT_INDEX ||
       cap == CAP_SHADER_VIEWPORT_INDEX_LAYER)
      return EXT_VIEWPORT_LAYER;
   if (cap >= CAP_STORAGE_BUFFER_8BIT && cap <= CAP_STORAGE_PUSH_CONSTANT_8)
      return EXT_8BIT;
   if (cap >= CAP_DENORM_PRESERVE && cap <= CAP_ROUNDING_MODE_RTZ)
      return EXT_FLOAT_CONTROLS;
   if (cap >= CAP_SHADER_NON_UNIFORM && cap <= CAP_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM)
      return EXT_DESCRIPTOR_INDEXING;
   if (cap == CAP_VULKAN_MEMORY_MODEL || cap == CAP_VULKAN_MEMORY_MODEL_DEVICE_SCOPE)
      return EXT_VMM;
   if (cap == CAP_PHYSICAL_STORAGE_BUFFER)
      return EXT_PHYSICAL_STORAGE;
   if (cap == CAP_DEMOTE_TO_HELPER)
      return EXT_DEMOTE;
   if (cap >= CAP_DOT_PRODUCT_INPUT_ALL && cap <= CAP_DOT_PRODUCT)
      return EXT_DOT_PRODUCT;
   return -1;
}

static int
image_operand_index(uint32_t op)
{
   switch (op) {
   case 87: case 88: case 91: case 92: case 95: case 98:
   case 305: case 306: case 309: case 310: case 313: case 320:
      return 5;
   case 89: case 90: case 93: case 94: case 96: case 97:
   case 307: case 308: case 311: case 312: case 314: case 315:
      return 6;
   case 99:
      return 4;
   default:
      return 0;
   }
}

static uint32_t
string_words(const uint32_t *w, uint32_t max)
{
   for (uint32_t i = 0; i < max; i++) {
      uint32_t v = w[i];
      if (!(v & 0xff) || !(v & 0xff00) || !(v & 0xff0000) || !(v & 0xff000000))
         return i + 1;
   }
   return max;
}

static bool
string_equals(const uint32_t *w, uint32_t max, const char *s)
{
   size_t len = strlen(s);
   if ((len + 4) / 4 > max)
      return false;
   return !memcmp(w, s, len) && ((const char *)w)[len] == 0;
}

static bool
string_prefix(const uint32_t *w, uint32_t max, const char *s)
{
   size_t len = strlen(s);
   return max * 4 > len && !memcmp(w, s, len);
}

struct spv_buf {
   uint32_t *d;
   size_t n, cap;
   bool oom;
};

static uint32_t *
buf_reserve(struct spv_buf *b, size_t words)
{
   if (b->n + words > b->cap) {
      size_t cap = MAX(b->cap * 2, b->n + words + 64);
      uint32_t *d = realloc(b->d, cap * sizeof(uint32_t));
      if (!d) {
         b->oom = true;
         return NULL;
      }
      b->d = d;
      b->cap = cap;
   }
   uint32_t *p = &b->d[b->n];
   b->n += words;
   return p;
}

static void
buf_put(struct spv_buf *b, const uint32_t *w, size_t words)
{
   uint32_t *p = buf_reserve(b, words);
   if (p)
      memcpy(p, w, words * sizeof(uint32_t));
}

static void
emit_string_inst(struct spv_buf *b, uint32_t opcode, const char *s)
{
   size_t len = strlen(s);
   uint32_t words = (uint32_t)(len / 4 + 1);
   uint32_t *p = buf_reserve(b, words + 1);
   if (!p)
      return;
   p[0] = ((words + 1) << 16) | opcode;
   memset(&p[1], 0, words * 4);
   memcpy(&p[1], s, len);
}

struct spv_info {
   const uint32_t *in;
   uint32_t bound;
   uint8_t *storage;
   uint16_t *type_op;
   uint32_t *def;
   uint32_t next_id;
};

static uint32_t
spv_constant_value(const struct spv_info *info, uint32_t id)
{
   if (id >= info->bound || !info->def[id])
      return 0;
   const uint32_t *w = &info->in[info->def[id]];
   return (w[0] & 0xffff) == OP_CONSTANT ? w[3] : 0;
}

static uint32_t
spv_copy_logical(struct spv_info *info, struct spv_buf *b, uint32_t dst_type,
                 uint32_t src_type, uint32_t src, uint32_t result)
{
   if (dst_type == src_type) {
      if (!result)
         return src;
      uint32_t w[4] = { (4u << 16) | OP_COPY_OBJECT, dst_type, result, src };
      buf_put(b, w, 4);
      return result;
   }
   if (dst_type >= info->bound || src_type >= info->bound ||
       !info->def[dst_type] || !info->def[src_type]) {
      b->oom = true;
      return 0;
   }
   const uint32_t *dt = &info->in[info->def[dst_type]];
   const uint32_t *st = &info->in[info->def[src_type]];
   uint32_t count;
   if ((dt[0] & 0xffff) == OP_TYPE_STRUCT)
      count = (dt[0] >> 16) - 2;
   else if ((dt[0] & 0xffff) == OP_TYPE_ARRAY)
      count = spv_constant_value(info, dt[3]);
   else {
      b->oom = true;
      return 0;
   }
   if (!count || count > 4096) {
      b->oom = true;
      return 0;
   }

   uint32_t *parts = malloc(count * sizeof(uint32_t));
   if (!parts) {
      b->oom = true;
      return 0;
   }
   for (uint32_t k = 0; k < count && !b->oom; k++) {
      bool is_struct = (dt[0] & 0xffff) == OP_TYPE_STRUCT;
      uint32_t dm = is_struct ? dt[2 + k] : dt[2];
      uint32_t sm = is_struct ? st[2 + k] : st[2];
      uint32_t e = info->next_id++;
      uint32_t w[5] = { (5u << 16) | OP_COMPOSITE_EXTRACT, sm, e, src, k };
      buf_put(b, w, 5);
      parts[k] = spv_copy_logical(info, b, dm, sm, e, 0);
   }
   if (!result)
      result = info->next_id++;
   uint32_t *p = buf_reserve(b, 3 + count);
   if (p) {
      p[0] = ((3 + count) << 16) | OP_COMPOSITE_CONSTRUCT;
      p[1] = dst_type;
      p[2] = result;
      memcpy(&p[3], parts, count * sizeof(uint32_t));
   }
   free(parts);
   return result;
}

enum wrapper_spirv_lower_result
wrapper_spirv_lower_to_1_3(const uint32_t *in, size_t words,
                           uint32_t **out_code, size_t *out_words)
{
   if (words < 5 || in[0] != SPV_MAGIC || in[1] <= SPV_1_3)
      return WRAPPER_SPIRV_UNCHANGED;

   struct spv_info info = {
      .in = in,
      .bound = in[3],
      .next_id = in[3],
   };
   uint32_t alloc = info.bound ? info.bound : 1;
   info.storage = calloc(alloc, 1);
   info.type_op = calloc(alloc, sizeof(uint16_t));
   info.def = calloc(alloc, sizeof(uint32_t));
   bool need[EXT_COUNT] = { false };
   bool have[EXT_COUNT] = { false };
   bool have_device_scope = false;
   bool ok = info.storage && info.type_op && info.def;

   for (size_t i = 5; ok && i < words;) {
      uint32_t op = in[i] & 0xffff, n = in[i] >> 16;
      if (!n || i + n > words) {
         ok = false;
         break;
      }
      const uint32_t *w = &in[i];
      switch (op) {
      case OP_CAPABILITY: {
         int e = cap_extension(w[1]);
         if (e >= 0)
            need[e] = true;
         if (w[1] == CAP_VULKAN_MEMORY_MODEL_DEVICE_SCOPE)
            have_device_scope = true;
         break;
      }
      case OP_EXTENSION:
         for (int e = 0; e < EXT_COUNT; e++)
            if (string_equals(&w[1], n - 1, ext_names[e]))
               have[e] = true;
         break;
      case OP_EXT_INST_IMPORT:
         if (string_prefix(&w[2], n - 2, "NonSemantic."))
            need[EXT_NON_SEMANTIC] = true;
         break;
      case OP_TYPE_ARRAY:
      case OP_TYPE_RUNTIME_ARRAY:
      case OP_TYPE_STRUCT:
      case OP_TYPE_POINTER:
         if (n > 1 && w[1] < info.bound) {
            info.type_op[w[1]] = (uint16_t)op;
            info.def[w[1]] = (uint32_t)i;
         }
         break;
      case OP_CONSTANT:
         if (n > 3 && w[2] < info.bound)
            info.def[w[2]] = (uint32_t)i;
         break;
      case OP_VARIABLE:
         if (n > 3 && w[2] < info.bound)
            info.storage[w[2]] = (uint8_t)(w[3] + 1);
         break;
      case OP_SELECT:
         if (n > 1 && w[1] < info.bound && info.type_op[w[1]])
            ok = false;
         break;
      case OP_PTR_EQUAL:
      case OP_PTR_NOT_EQUAL:
      case OP_PTR_DIFF:
         ok = false;
         break;
      default:
         break;
      }
      i += n;
   }

   struct spv_buf b = { 0 };
   if (ok && !buf_reserve(&b, 5))
      ok = false;
   if (ok) {
      memcpy(b.d, in, 5 * sizeof(uint32_t));
      b.d[1] = SPV_1_3;
   }

   bool emitted_viewport_layer = false;
   bool extensions_written = false;

   for (size_t i = 5; ok && !b.oom && i < words;) {
      uint32_t op = in[i] & 0xffff, n = in[i] >> 16;
      const uint32_t *w = &in[i];
      i += n;

      if (!extensions_written && op != OP_CAPABILITY) {
         for (int e = 0; e < EXT_COUNT; e++)
            if (need[e] && !have[e])
               emit_string_inst(&b, OP_EXTENSION, ext_names[e]);
         extensions_written = true;
      }

      switch (op) {
      case OP_EXTENSION:
         if (string_equals(&w[1], n - 1, "SPV_KHR_terminate_invocation"))
            continue;
         break;
      case OP_CAPABILITY:
         if (w[1] == CAP_VULKAN_MEMORY_MODEL && !have_device_scope) {
            uint32_t c[4] = { (2u << 16) | OP_CAPABILITY, CAP_VULKAN_MEMORY_MODEL,
                              (2u << 16) | OP_CAPABILITY, CAP_VULKAN_MEMORY_MODEL_DEVICE_SCOPE };
            buf_put(&b, c, 4);
            continue;
         }
         if (w[1] == CAP_SHADER_LAYER || w[1] == CAP_SHADER_VIEWPORT_INDEX ||
             w[1] == CAP_SHADER_VIEWPORT_INDEX_LAYER) {
            if (!emitted_viewport_layer) {
               uint32_t c[2] = { (2u << 16) | OP_CAPABILITY, CAP_SHADER_VIEWPORT_INDEX_LAYER };
               buf_put(&b, c, 2);
               emitted_viewport_layer = true;
            }
            continue;
         }
         break;
      case OP_ENTRY_POINT: {
         uint32_t fixed = 3 + string_words(&w[3], n - 3);
         size_t start = b.n;
         buf_put(&b, w, fixed);
         for (uint32_t k = fixed; k < n; k++) {
            uint32_t id = w[k];
            uint32_t sc = id < info.bound ? info.storage[id] : 0;
            if (sc == 0 || sc == 1 + 1 || sc == 3 + 1)
               buf_put(&b, &id, 1);
         }
         if (!b.oom)
            b.d[start] = ((uint32_t)(b.n - start) << 16) | OP_ENTRY_POINT;
         continue;
      }
      case OP_DECORATE:
         if (n >= 3 && (w[2] == 4469 || w[2] == 4470))
            continue;
         if (n >= 3 && w[2] == 24 && w[1] < info.bound &&
             (info.storage[w[1]] == 6 + 1 || info.storage[w[1]] == 7 + 1))
            continue;
         break;
      case OP_MEMBER_DECORATE:
         if (n >= 4 && (w[3] == 4469 || w[3] == 4470))
            continue;
         break;
      case OP_TERMINATE_INVOCATION: {
         uint32_t k = (1u << 16) | OP_KILL;
         buf_put(&b, &k, 1);
         continue;
      }
      case OP_COPY_LOGICAL: {
         uint32_t src_type = 0;
         if (n == 4 && w[3] < info.bound)
            for (size_t j = 5; j < words; j += in[j] >> 16) {
               uint32_t jop = in[j] & 0xffff;
               if ((in[j] >> 16) > 2 && in[j + 2] == w[3] &&
                   jop != OP_DECORATE && jop != OP_MEMBER_DECORATE &&
                   jop != 5 && jop != 6) {
                  src_type = in[j + 1];
                  break;
               }
               if (!(in[j] >> 16))
                  break;
            }
         if (!src_type) {
            ok = false;
            continue;
         }
         spv_copy_logical(&info, &b, w[1], src_type, w[3], w[2]);
         continue;
      }
      default: {
         int idx = image_operand_index(op);
         if (idx && n > (uint32_t)idx) {
            size_t at = b.n;
            buf_put(&b, w, n);
            if (!b.oom)
               b.d[at + idx] &= ~(0x1000u | 0x2000u | 0x4000u);
            continue;
         }
         break;
      }
      }

      buf_put(&b, w, n);
   }

   free(info.storage);
   free(info.type_op);
   free(info.def);

   if (!ok || b.oom) {
      free(b.d);
      return WRAPPER_SPIRV_FAILED;
   }
   b.d[3] = info.next_id;
   *out_code = b.d;
   *out_words = b.n;
   return WRAPPER_SPIRV_LOWERED;
}
