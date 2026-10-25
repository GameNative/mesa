#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "astcenc/astcenc.h"
#include "wrapper_astcenc.h"

namespace {

const unsigned max_threads = 8;

struct slot {
   bool used;
   int block, kind, srgb;
   unsigned threads;
   float quality;
   astcenc_context *ctx;
   pthread_mutex_t lock;
};

slot slots[32];
pthread_mutex_t slots_lock = PTHREAD_MUTEX_INITIALIZER;

float
preset()
{
   const char *e = getenv("WRAPPER_ASTC_PRESET");
   if (!e)
      return ASTCENC_PRE_FASTEST;
   if (!strcmp(e, "fast"))
      return ASTCENC_PRE_FAST;
   if (!strcmp(e, "medium"))
      return ASTCENC_PRE_MEDIUM;
   if (!strcmp(e, "thorough"))
      return ASTCENC_PRE_THOROUGH;
   return ASTCENC_PRE_FASTEST;
}

unsigned
core_threads()
{
   long n = sysconf(_SC_NPROCESSORS_CONF);
   return n < 1 ? 1 : n > (long)max_threads ? max_threads : (unsigned)n;
}

slot *
get_slot(int block, int kind, int srgb, unsigned threads, float quality)
{
   pthread_mutex_lock(&slots_lock);
   slot *found = nullptr, *free_slot = nullptr;
   for (slot &s : slots) {
      if (s.used && s.block == block && s.kind == kind && s.srgb == srgb && s.threads == threads &&
          s.quality == quality) {
         found = &s;
         break;
      }
      if (!s.used && !free_slot)
         free_slot = &s;
   }
   if (!found && free_slot) {
      astcenc_profile prf = kind == WRAPPER_ASTC_HDR_RGB ? ASTCENC_PRF_HDR_RGB_LDR_A
                            : srgb ? ASTCENC_PRF_LDR_SRGB : ASTCENC_PRF_LDR;
      astcenc_config cfg;
      if (astcenc_config_init(prf, block, block, 1, quality, 0, &cfg) == ASTCENC_SUCCESS) {
         if (kind == WRAPPER_ASTC_L) {
            cfg.cw_g_weight = 0.0f;
            cfg.cw_b_weight = 0.0f;
            cfg.cw_a_weight = 0.0f;
         } else if (kind == WRAPPER_ASTC_LA) {
            cfg.cw_g_weight = 0.0f;
            cfg.cw_b_weight = 0.0f;
         }
         astcenc_context *ctx = nullptr;
         if (astcenc_context_alloc(&cfg, threads, &ctx, nullptr) == ASTCENC_SUCCESS) {
            free_slot->used = true;
            free_slot->block = block;
            free_slot->kind = kind;
            free_slot->srgb = srgb;
            free_slot->threads = threads;
            free_slot->quality = quality;
            free_slot->ctx = ctx;
            pthread_mutex_init(&free_slot->lock, nullptr);
            found = free_slot;
         }
      }
   }
   pthread_mutex_unlock(&slots_lock);
   return found;
}

struct job {
   astcenc_context *ctx;
   astcenc_image *image;
   const astcenc_swizzle *swizzle;
   uint8_t *out;
   size_t len;
   unsigned index;
   astcenc_error result;
};

void *
run_job(void *arg)
{
   job *j = static_cast<job *>(arg);
   j->result = astcenc_compress_image(j->ctx, j->image, j->swizzle, j->out, j->len, j->index);
   return nullptr;
}

} // namespace

extern "C" int
wrapper_astcenc_compress(const void *texels, int w, int h, int block,
                         enum wrapper_astc_kind kind, int srgb, uint8_t *out, size_t len)
{
   return wrapper_astcenc_compress_threads(texels, w, h, block, kind, srgb, out, len, 0, -1.0f);
}

extern "C" int
wrapper_astcenc_compress_threads(const void *texels, int w, int h, int block,
                                 enum wrapper_astc_kind kind, int srgb, uint8_t *out,
                                 size_t len, unsigned threads, float quality)
{
   size_t blocks = (size_t)((w + block - 1) / block) * ((h + block - 1) / block);
   if (blocks < 256)
      threads = 1;
   else if (!threads)
      threads = core_threads();
   threads = threads > max_threads ? max_threads : threads;
   slot *s = get_slot(block, kind, srgb, threads, quality < 0.0f ? preset() : quality);
   if (!s)
      return -1;

   astcenc_swizzle swz = { ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
   if (kind == WRAPPER_ASTC_L)
      swz = { ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_1 };
   else if (kind == WRAPPER_ASTC_LA)
      swz = { ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_G };
   else if (kind == WRAPPER_ASTC_HDR_RGB)
      swz = { ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_1 };

   void *slice = const_cast<void *>(texels);
   astcenc_image image;
   image.dim_x = w;
   image.dim_y = h;
   image.dim_z = 1;
   image.data_type = kind == WRAPPER_ASTC_HDR_RGB ? ASTCENC_TYPE_F16 : ASTCENC_TYPE_U8;
   image.data = &slice;

   pthread_mutex_lock(&s->lock);
   job jobs[max_threads];
   pthread_t tids[max_threads];
   bool started[max_threads] = {};
   for (unsigned i = 0; i < threads; i++) {
      jobs[i] = { s->ctx, &image, &swz, out, len, i, ASTCENC_SUCCESS };
      if (i > 0)
         started[i] = pthread_create(&tids[i], nullptr, run_job, &jobs[i]) == 0;
   }
   run_job(&jobs[0]);
   for (unsigned i = 1; i < threads; i++) {
      if (started[i])
         pthread_join(tids[i], nullptr);
      else
         run_job(&jobs[i]);
   }
   astcenc_compress_reset(s->ctx);
   pthread_mutex_unlock(&s->lock);

   for (unsigned i = 0; i < threads; i++)
      if (jobs[i].result != ASTCENC_SUCCESS)
         return -1;
   return 0;
}
