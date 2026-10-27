/*
 * Disk cache of the background stage's ASTC results.
 *
 * An entry holds the astcenc encode of one whole mip level, keyed by the BC
 * source blocks, the BC format, the size and the stored ASTC format, under a
 * directory named after the encoder settings, so a wrapper with another
 * encoder never reads it. A GPU upload of a level that has an entry copies
 * the cached blocks into the image and skips both encodes.
 *
 * WRAPPER_USE_BCN_CACHE turns it on, WRAPPER_BCN_CACHE_MB caps its size
 * (4096; the oldest entries go first) and WRAPPER_BCN_CACHE_MIN_FREE_MB is
 * the free space a write leaves on the filesystem (2048). WRAPPER_LOG_LEVEL
 * with bcn logs every lookup and write.
 */

#include "wrapper_bcdec.h"
#include "wrapper_bcn_gpu.h"
#include "wrapper_private.h"
#include "wrapper_log.h"

#define XXH_STATIC_LINKING_ONLY
#include "util/xxhash.h"

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>

#define BCN_CACHE_ENCODER "e1"
#define BCN_CACHE_MAGIC 0x48434e42u
#define BCN_CACHE_MIN_TEXELS (128u * 128u)

struct bcn_cache_header {
   uint32_t magic;
   uint32_t size;
   uint64_t hash;
};

static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static char *cache_dir;
static int64_t cache_bytes = -1;
static uint64_t cache_hits, cache_misses, cache_writes, cache_written, cache_skipped;

static uint64_t
env_mb(const char *name, uint64_t def)
{
   const char *e = getenv(name);
   return (uint64_t)(e && atoll(e) > 0 ? atoll(e) : (long long)def) << 20;
}

bool
wrapper_bcn_cache_on(void)
{
   return bcn_cache_enabled();
}

/* the directory of this encoder's entries, made on first use */
static const char *
cache_path(void)
{
   if (!cache_dir) {
      const char *root = getenv("WRAPPER_CACHE_PATH");
      char *dir = NULL;
      if (asprintf(&dir, "%s/bcn_hq_%s_%s", root ? root : "/data/data/app.gamenative/files/imagefs/usr/cache",
                   BCN_CACHE_ENCODER, wrapper_bcn_hq_settings()) < 0)
         return NULL;
      if (root)
         mkdir(root, 0700);
      if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
         free(dir);
         return NULL;
      }
      cache_dir = dir;
   }
   return cache_dir;
}

static const char *
bc_tag(VkFormat f)
{
   switch (f) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return "bc1";
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return "bc1s";
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return "bc1a";
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return "bc1as";
   case VK_FORMAT_BC2_UNORM_BLOCK: return "bc2";
   case VK_FORMAT_BC2_SRGB_BLOCK: return "bc2s";
   case VK_FORMAT_BC3_UNORM_BLOCK: return "bc3";
   case VK_FORMAT_BC3_SRGB_BLOCK: return "bc3s";
   case VK_FORMAT_BC4_UNORM_BLOCK: return "bc4";
   case VK_FORMAT_BC5_UNORM_BLOCK: return "bc5";
   case VK_FORMAT_BC6H_UFLOAT_BLOCK: return "bc6hu";
   case VK_FORMAT_BC6H_SFLOAT_BLOCK: return "bc6hs";
   case VK_FORMAT_BC7_UNORM_BLOCK: return "bc7";
   case VK_FORMAT_BC7_SRGB_BLOCK: return "bc7s";
   default: return NULL;
   }
}

static const char *
astc_tag(VkFormat stored)
{
   if (is_astc_hdr_4x4(stored))
      return "h4";
   if (is_astc_4x4(stored))
      return "a4";
   if (is_astc_6x6(stored))
      return "a6";
   if (is_astc_8x8(stored))
      return "a8";
   return NULL;
}

/* key of tightly or loosely packed BC rows: rows of row_bytes, stride apart */
static char *
entry_name(VkFormat bc, VkFormat stored, uint32_t w, uint32_t h, const uint8_t *src,
           size_t row_bytes, size_t stride, uint32_t rows)
{
   const char *bt = bc_tag(bc), *at = astc_tag(stored), *dir = cache_path();
   if (!bt || !at || !dir)
      return NULL;
   XXH64_state_t *st = XXH64_createState();
   if (!st)
      return NULL;
   XXH64_reset(st, 0);
   for (uint32_t y = 0; y < rows; y++)
      XXH64_update(st, src + (size_t)y * stride, row_bytes);
   uint64_t key = XXH64_digest(st);
   XXH64_freeState(st);
   char *name = NULL;
   if (asprintf(&name, "%s/%s_%ux%u_%016llx.%s", dir, bt, w, h, (unsigned long long)key, at) < 0)
      return NULL;
   return name;
}

static void
cache_log(const char *what)
{
   wrapper_diag_append("[BCN] cache %s: hits %llu, misses %llu, writes %llu (%.1f MB), "
                       "skipped %llu, size %.1f MB\n", what, (unsigned long long)cache_hits,
                       (unsigned long long)cache_misses, (unsigned long long)cache_writes,
                       cache_written / 1048576.0, (unsigned long long)cache_skipped,
                       cache_bytes < 0 ? 0.0 : cache_bytes / 1048576.0);
}

/* every 64th event, and the first hit and the first write */
static void
cache_note(uint64_t count, const char *what)
{
   if (count == 1 || (cache_hits + cache_misses + cache_writes) % 64 == 0)
      cache_log(what);
}

bool
wrapper_bcn_cache_load(VkFormat bc, VkFormat stored, uint32_t w, uint32_t h,
                       const uint8_t *src, size_t row_bytes, size_t stride, uint32_t rows,
                       uint8_t *out, size_t out_len)
{
   if (!wrapper_bcn_cache_on() || (uint64_t)w * h < BCN_CACHE_MIN_TEXELS)
      return false;
   char *name = entry_name(bc, stored, w, h, src, row_bytes, stride, rows);
   if (!name)
      return false;
   bool hit = false;
   FILE *f = fopen(name, "rb");
   if (f) {
      struct bcn_cache_header hd;
      hit = fread(&hd, sizeof(hd), 1, f) == 1 && hd.magic == BCN_CACHE_MAGIC &&
            hd.size == out_len && fread(out, 1, out_len, f) == out_len &&
            XXH64(out, out_len, 0) == hd.hash;
      fclose(f);
      if (hit)
         utime(name, NULL);
      else
         unlink(name);
   }
   free(name);
   if (get_wrapper_log_level("bcn"))
      wrapper_diag_append("[BCN] cache %s %s %ux%u -> %s\n", hit ? "hit" : "miss", bc_tag(bc), w,
                          h, astc_tag(stored));
   pthread_mutex_lock(&cache_lock);
   if (hit)
      cache_note(++cache_hits, "hit");
   else
      cache_note(++cache_misses, "miss");
   pthread_mutex_unlock(&cache_lock);
   return hit;
}

struct cache_file {
   char *path;
   time_t mtime;
   off_t size;
};

static int
cache_file_cmp(const void *a, const void *b)
{
   const struct cache_file *x = a, *y = b;
   return x->mtime < y->mtime ? -1 : x->mtime > y->mtime;
}

/* size of the directory; with evict, removes the oldest entries until
 * keep bytes are left. Called with cache_lock held. */
static int64_t
cache_scan(const char *dir, int64_t keep, bool evict)
{
   DIR *d = opendir(dir);
   if (!d)
      return 0;
   struct cache_file *files = NULL;
   size_t n = 0, cap = 0;
   int64_t total = 0;
   struct dirent *e;
   while ((e = readdir(d))) {
      if (e->d_name[0] == '.')
         continue;
      char *p = NULL;
      struct stat sb;
      if (asprintf(&p, "%s/%s", dir, e->d_name) < 0)
         continue;
      if (stat(p, &sb) != 0 || !S_ISREG(sb.st_mode)) {
         free(p);
         continue;
      }
      total += sb.st_size;
      if (!evict) {
         free(p);
         continue;
      }
      if (n == cap) {
         cap = cap ? cap * 2 : 256;
         struct cache_file *nf = realloc(files, cap * sizeof(*files));
         if (!nf) {
            free(p);
            break;
         }
         files = nf;
      }
      files[n++] = (struct cache_file){ p, sb.st_mtime, sb.st_size };
   }
   closedir(d);
   if (evict) {
      qsort(files, n, sizeof(*files), cache_file_cmp);
      for (size_t i = 0; i < n; i++) {
         if (total > keep && unlink(files[i].path) == 0)
            total -= files[i].size;
         free(files[i].path);
      }
      free(files);
   }
   return total;
}

void
wrapper_bcn_cache_store(VkFormat bc, VkFormat stored, uint32_t w, uint32_t h,
                        const uint8_t *src, size_t row_bytes, uint32_t rows,
                        const uint8_t *data, size_t len)
{
   if (!wrapper_bcn_cache_on() || (uint64_t)w * h < BCN_CACHE_MIN_TEXELS)
      return;
   char *name = entry_name(bc, stored, w, h, src, row_bytes, row_bytes, rows);
   if (!name)
      return;
   const char *dir = cache_path();
   uint64_t cap = env_mb("WRAPPER_BCN_CACHE_MB", 4096);
   uint64_t min_free = env_mb("WRAPPER_BCN_CACHE_MIN_FREE_MB", 2048);
   size_t need = len + sizeof(struct bcn_cache_header);

   pthread_mutex_lock(&cache_lock);
   struct statvfs vfs;
   bool room = dir && statvfs(dir, &vfs) == 0 &&
               (uint64_t)vfs.f_bavail * vfs.f_frsize >= min_free + need;
   if (room) {
      if (cache_bytes < 0)
         cache_bytes = cache_scan(dir, 0, false);
      if ((uint64_t)cache_bytes + need > cap)
         cache_bytes = cache_scan(dir, (int64_t)(cap * 9 / 10) - (int64_t)need, true);
      room = (uint64_t)cache_bytes + need <= cap;
   }
   if (!room) {
      cache_note(++cache_skipped, "write skipped");
      pthread_mutex_unlock(&cache_lock);
      free(name);
      return;
   }
   pthread_mutex_unlock(&cache_lock);

   char *tmp = NULL;
   bool ok = false;
   if (asprintf(&tmp, "%s.%d.tmp", name, (int)getpid()) >= 0) {
      FILE *f = fopen(tmp, "wb");
      if (f) {
         struct bcn_cache_header hd = { BCN_CACHE_MAGIC, (uint32_t)len, XXH64(data, len, 0) };
         ok = fwrite(&hd, sizeof(hd), 1, f) == 1 && fwrite(data, 1, len, f) == len;
         ok = fclose(f) == 0 && ok && rename(tmp, name) == 0;
         if (!ok)
            unlink(tmp);
      }
      free(tmp);
   }
   free(name);
   if (get_wrapper_log_level("bcn"))
      wrapper_diag_append("[BCN] cache write %s %ux%u -> %s: %s\n", bc_tag(bc), w, h,
                          astc_tag(stored), ok ? "ok" : "failed");
   pthread_mutex_lock(&cache_lock);
   if (ok) {
      cache_bytes += need;
      cache_written += need;
      cache_note(++cache_writes, "write");
   }
   pthread_mutex_unlock(&cache_lock);
}

void
wrapper_bcn_cache_report(void)
{
   if (!wrapper_bcn_cache_on())
      return;
   pthread_mutex_lock(&cache_lock);
   if (cache_hits || cache_misses || cache_writes || cache_skipped)
      cache_log("totals");
   pthread_mutex_unlock(&cache_lock);
}
