#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "wrapper_bcn_gpu.h"
#include "wrapper_bcdec.h"
#include "wrapper_log.h"
#include "util/macros.h"

/* Second stage of a GPU upload to ASTC. The GPU encoder makes the texture
 * usable at once; the BC blocks of the upload are also copied, encoded again
 * by astcenc on a background thread, and the result is copied over the same
 * texels on a later submit to the queue the upload went to, between two
 * barriers from and back to the layout the image was last moved to. A later
 * write to the same texels, the image's destruction or the command buffer
 * being reset before it is submitted drops the job, and so does a barrier
 * moving the image to another queue family or a submit of a command buffer
 * that moved it to another queue: the copy is only ordered against work on
 * its own queue.
 *
 * The worker runs at the lowest priority (SCHED_IDLE, nice 19), encodes in
 * strips of 96 rows, waits between strips while CPU BCn work the application
 * waits for is running, and is held to WRAPPER_BCN_HQ_MTEXPS Mtexel per
 * second (4).
 * WRAPPER_BCN_HQ=0 turns it off, WRAPPER_BCN_HQ_THREADS sets the encoder
 * threads (2), WRAPPER_BCN_HQ_QUALITY the astcenc quality (60, its medium
 * preset), WRAPPER_BCN_HQ_MB caps the blocks waiting (256). */

enum { HQ_QUEUED, HQ_ENCODING, HQ_READY, HQ_DROPPED };

struct bcn_hq_job {
   struct list_head link;
   struct wrapper_image *wi;
   struct wrapper_command_buffer *wcb;
   struct wrapper_queue *queue;
   VkFormat format;
   VkImageSubresourceLayers sub;
   VkOffset3D offset;
   VkExtent3D extent;
   uint32_t block;
   size_t layer_in, layer_out;
   uint8_t *in;
   uint8_t *out;
   int state;
   double t_capture, encode_ms;
};

struct bcn_hq_batch {
   struct list_head link;
   VkFence fence;
   VkCommandPool pool;
   VkCommandBuffer cb;
   VkBuffer buffer;
   VkDeviceMemory memory;
};

struct bcn_hq {
   pthread_mutex_t lock;
   pthread_cond_t wake;
   pthread_t thread;
   bool stop;
   struct list_head jobs;
   struct list_head batches;
   size_t bytes;
   unsigned threads;
   float quality;
   double mtexps;
   unsigned drops;
   size_t cap;
   VkCommandPool pools[16];
};

static const char *
hq_format_name(VkFormat f)
{
   switch (f) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return "bc1";
   case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: return "bc2";
   case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: return "bc3";
   case VK_FORMAT_BC4_UNORM_BLOCK: return "bc4";
   case VK_FORMAT_BC5_UNORM_BLOCK: return "bc5";
   case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK: return "bc6h";
   default: return "bc7";
   }
}

uint32_t wrapper_bcn_fg_busy;

static double
now_ms(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int
hq_enabled(void)
{
   static int on = -1;
   if (on == -1) {
      const char *e = getenv("WRAPPER_BCN_HQ");
      on = !(e && !atoi(e));
   }
   return on;
}

static void
hq_free_job(struct bcn_hq *hq, struct bcn_hq_job *j)
{
   list_del(&j->link);
   hq->bytes -= j->layer_in * j->sub.layerCount;
   if (j->wi)
      j->wi->bcn_hq_jobs--;
   free(j->in);
   free(j->out);
   free(j);
}

static void
hq_drop(struct wrapper_device *device, struct bcn_hq *hq, struct bcn_hq_job *j);

static void
hq_drop_logged(struct wrapper_device *device, struct bcn_hq *hq, struct bcn_hq_job *j,
               const char *why)
{
   if (hq->drops++ < 16 || (hq->drops & (hq->drops - 1)) == 0)
      wrapper_diag_append("[BCN] hq drop %s %ux%u mip %u: %s (%u drops)\n",
                          hq_format_name(j->format), j->extent.width, j->extent.height,
                          j->sub.mipLevel, why, hq->drops);
   hq_drop(device, hq, j);
}

static void
hq_drop(struct wrapper_device *device, struct bcn_hq *hq, struct bcn_hq_job *j)
{
   if (j->wi)
      j->wi->bcn_hq_jobs--;
   j->wi = NULL;
   if (j->state != HQ_ENCODING) {
      hq_free_job(hq, j);
      __atomic_sub_fetch(&device->bcn_hq_active, 1, __ATOMIC_RELAXED);
   } else {
      j->state = HQ_DROPPED;
   }
}

static void *
hq_worker(void *arg)
{
   struct wrapper_device *device = arg;
   struct bcn_hq *hq = device->bcn_hq;
#ifdef SCHED_IDLE
   struct sched_param sp = { 0 };
   sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
   setpriority(PRIO_PROCESS, 0, 19);
   pthread_mutex_lock(&hq->lock);
   while (!hq->stop) {
      struct bcn_hq_job *job = NULL;
      list_for_each_entry(struct bcn_hq_job, j, &hq->jobs, link) {
         if (j->state == HQ_QUEUED) {
            job = j;
            break;
         }
      }
      if (!job) {
         pthread_cond_wait(&hq->wake, &hq->lock);
         continue;
      }
      job->state = HQ_ENCODING;
      pthread_mutex_unlock(&hq->lock);
      double t0 = now_ms();
      bool ok = true;
      uint32_t w = job->extent.width, h = job->extent.height;
      uint32_t bb = (uint32_t)(job->layer_in / (((w + 3) / 4) * ((h + 3) / 4)));
      size_t out_row = (size_t)((w + job->block - 1) / job->block) * 16;
      for (uint32_t l = 0; l < job->sub.layerCount && ok; l++) {
         for (uint32_t y = 0; y < h && ok; y += 96) {
            while (__atomic_load_n(&wrapper_bcn_fg_busy, __ATOMIC_RELAXED) &&
                   !__atomic_load_n(&hq->stop, __ATOMIC_RELAXED))
               usleep(2000);
            uint32_t sh = MIN2(96u, h - y);
            double s0 = now_ms();
            ok = wrapper_bcn_astc_encode_cpu(job->in + l * job->layer_in +
                                                (size_t)(y / 4) * ((w + 3) / 4) * bb,
                                             w, sh, job->format,
                                             job->out + l * job->layer_out +
                                                (size_t)(y / job->block) * out_row,
                                             job->layer_out - (size_t)(y / job->block) * out_row,
                                             hq->threads, hq->quality) == 0;
            double budget = (double)w * sh / (hq->mtexps * 1000.0);
            double spent = now_ms() - s0;
            if (spent < budget)
               usleep((useconds_t)((budget - spent) * 1000.0));
         }
      }
      double t1 = now_ms();
      pthread_mutex_lock(&hq->lock);
      job->encode_ms = t1 - t0;
      free(job->in);
      job->in = NULL;
      if (job->state == HQ_DROPPED || !ok) {
         if (job->wi)
            job->wi->bcn_hq_jobs--;
         job->wi = NULL;
         hq_free_job(hq, job);
         __atomic_sub_fetch(&device->bcn_hq_active, 1, __ATOMIC_RELAXED);
      } else {
         job->state = HQ_READY;
      }
   }
   pthread_mutex_unlock(&hq->lock);
   return NULL;
}

static struct bcn_hq *
hq_get(struct wrapper_device *device)
{
   simple_mtx_lock(&device->bcn_gpu_mutex);
   if (!device->bcn_hq) {
      struct bcn_hq *hq = calloc(1, sizeof(*hq));
      if (hq) {
         pthread_mutex_init(&hq->lock, NULL);
         pthread_cond_init(&hq->wake, NULL);
         list_inithead(&hq->jobs);
         list_inithead(&hq->batches);
         const char *t = getenv("WRAPPER_BCN_HQ_THREADS");
         const char *mb = getenv("WRAPPER_BCN_HQ_MB");
         const char *q = getenv("WRAPPER_BCN_HQ_QUALITY");
         hq->threads = t && atoi(t) > 0 ? atoi(t) : 2;
         hq->quality = q ? (float)atof(q) : 60.0f;
         const char *r = getenv("WRAPPER_BCN_HQ_MTEXPS");
         hq->mtexps = r && atof(r) > 0.0 ? atof(r) : 4.0;
         hq->cap = (size_t)(mb && atoi(mb) > 0 ? atoi(mb) : 256) << 20;
         device->bcn_hq = hq;
         if (pthread_create(&hq->thread, NULL, hq_worker, device) != 0) {
            device->bcn_hq = NULL;
            free(hq);
         }
      }
   }
   simple_mtx_unlock(&device->bcn_gpu_mutex);
   return device->bcn_hq;
}

static bool
hq_overlap(const struct bcn_hq_job *j, const VkImageSubresourceLayers *sub,
           VkOffset3D off, VkExtent3D ext)
{
   if (j->sub.mipLevel != sub->mipLevel ||
       j->sub.baseArrayLayer >= sub->baseArrayLayer + sub->layerCount ||
       sub->baseArrayLayer >= j->sub.baseArrayLayer + j->sub.layerCount)
      return false;
   return j->offset.x < off.x + (int32_t)ext.width && off.x < j->offset.x + (int32_t)j->extent.width &&
          j->offset.y < off.y + (int32_t)ext.height && off.y < j->offset.y + (int32_t)j->extent.height;
}

void
wrapper_bcn_hq_write(struct wrapper_device *device, struct wrapper_image *wi,
                     const VkImageSubresourceLayers *sub, VkOffset3D offset, VkExtent3D extent)
{
   if (!wi || !wi->bcn_hq_jobs || !device->bcn_hq)
      return;
   struct bcn_hq *hq = device->bcn_hq;
   pthread_mutex_lock(&hq->lock);
   list_for_each_entry_safe(struct bcn_hq_job, j, &hq->jobs, link)
      if (j->wi == wi && (!sub || hq_overlap(j, sub, offset, extent)))
         hq_drop(device, hq, j);
   pthread_mutex_unlock(&hq->lock);
}

void
wrapper_bcn_hq_capture(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       struct wrapper_buffer *wb, VkImageLayout layout,
                       const VkBufferImageCopy *c, uint32_t block)
{
   struct wrapper_device *device = wcb->device;
   VkFormat format = wi->info.format;
   wrapper_bcn_hq_write(device, wi, &c->imageSubresource, c->imageOffset, c->imageExtent);
   if (!hq_enabled() || !wb || wi->mip_drop || wcb->secondary ||
       wi->info.sharingMode != VK_SHARING_MODE_EXCLUSIVE)
      return;
   struct bcn_hq *hq = hq_get(device);
   if (!hq)
      return;

   uint32_t w = c->imageExtent.width, h = c->imageExtent.height;
   uint32_t layers = c->imageSubresource.layerCount;
   if (layers == VK_REMAINING_ARRAY_LAYERS)
      layers = wi->info.arrayLayers - c->imageSubresource.baseArrayLayer;
   uint32_t bb = format == VK_FORMAT_BC1_RGB_UNORM_BLOCK || format == VK_FORMAT_BC1_RGB_SRGB_BLOCK ||
                 format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK ||
                 format == VK_FORMAT_BC4_UNORM_BLOCK ? 8 : 16;
   uint32_t nbx = (w + 3) / 4, nby = (h + 3) / 4;
   uint32_t row_len = c->bufferRowLength ? c->bufferRowLength : w;
   uint32_t img_h = c->bufferImageHeight ? c->bufferImageHeight : h;
   size_t src_row = (size_t)((row_len + 3) / 4) * bb;
   size_t src_slice = src_row * ((img_h + 3) / 4);
   size_t tight = (size_t)nbx * bb;
   size_t layer_in = tight * nby;
   size_t layer_out = (size_t)((w + block - 1) / block) * ((h + block - 1) / block) * 16;

   pthread_mutex_lock(&hq->lock);
   bool room = hq->bytes + layer_in * layers <= hq->cap;
   pthread_mutex_unlock(&hq->lock);
   if (!room)
      return;

   simple_mtx_lock(&device->resource_mutex);
   if (!wb->is_mapped) {
      if (device->dispatch_table.MapMemory(device->dispatch_handle, wb->memory, wb->offset,
                                           wb->size, 0, &wb->mapped_address) != VK_SUCCESS) {
         simple_mtx_unlock(&device->resource_mutex);
         return;
      }
      wb->is_mapped = 1;
   }
   struct bcn_hq_job *j = calloc(1, sizeof(*j));
   uint8_t *in = malloc(layer_in * layers);
   uint8_t *out = malloc(layer_out * layers);
   if (!j || !in || !out || c->bufferOffset + src_slice * (layers - 1) + src_row * (nby - 1) + tight >
                            wb->size) {
      simple_mtx_unlock(&device->resource_mutex);
      free(j);
      free(in);
      free(out);
      return;
   }
   for (uint32_t l = 0; l < layers; l++)
      for (uint32_t y = 0; y < nby; y++)
         memcpy(in + l * layer_in + y * tight,
                (const uint8_t *)wb->mapped_address + c->bufferOffset + l * src_slice + y * src_row,
                tight);
   simple_mtx_unlock(&device->resource_mutex);

   j->wi = wi;
   j->wcb = wcb;
   j->format = format;
   j->sub = c->imageSubresource;
   j->sub.layerCount = layers;
   j->offset = c->imageOffset;
   j->extent = c->imageExtent;
   j->block = block;
   j->layer_in = layer_in;
   j->layer_out = layer_out;
   j->in = in;
   j->out = out;
   j->t_capture = now_ms();
   pthread_mutex_lock(&hq->lock);
   wi->bcn_hq_jobs++;
   wi->bcn_layout = layout;
   hq->bytes += layer_in * layers;
   list_addtail(&j->link, &hq->jobs);
   __atomic_add_fetch(&device->bcn_hq_active, 1, __ATOMIC_RELAXED);
   pthread_cond_signal(&hq->wake);
   pthread_mutex_unlock(&hq->lock);
}

void
wrapper_bcn_hq_barrier(struct wrapper_command_buffer *wcb, VkImage image, VkImageLayout old_layout,
                       VkImageLayout new_layout, uint32_t src_family, uint32_t dst_family)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_image *wi = get_wrapper_image_from_handle(device, image);
   if (!wi || !wi->bcn_hq_jobs || !device->bcn_hq)
      return;
   struct bcn_hq *hq = device->bcn_hq;
   pthread_mutex_lock(&hq->lock);
   if (src_family != dst_family && src_family != VK_QUEUE_FAMILY_IGNORED &&
       dst_family != VK_QUEUE_FAMILY_IGNORED) {
      list_for_each_entry_safe(struct bcn_hq_job, j, &hq->jobs, link)
         if (j->wi == wi)
            hq_drop_logged(device, hq, j, "queue family transfer");
   } else {
      if (old_layout != new_layout)
         wi->bcn_layout = new_layout;
      wi->bcn_touch = wcb;
   }
   pthread_mutex_unlock(&hq->lock);
}

void
wrapper_bcn_hq_image_destroy(struct wrapper_device *device, struct wrapper_image *wi)
{
   wrapper_bcn_hq_write(device, wi, NULL, (VkOffset3D){ 0 }, (VkExtent3D){ 0 });
}

void
wrapper_bcn_hq_cmd_reset(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   if (!__atomic_load_n(&device->bcn_hq_active, __ATOMIC_RELAXED) || !device->bcn_hq)
      return;
   struct bcn_hq *hq = device->bcn_hq;
   pthread_mutex_lock(&hq->lock);
   list_for_each_entry_safe(struct bcn_hq_job, j, &hq->jobs, link)
      if (j->wcb == wcb)
         hq_drop(device, hq, j);
   pthread_mutex_unlock(&hq->lock);
}

void
wrapper_bcn_hq_submitted(struct wrapper_queue *queue, struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = queue->device;
   if (!__atomic_load_n(&device->bcn_hq_active, __ATOMIC_RELAXED) || !device->bcn_hq)
      return;
   struct bcn_hq *hq = device->bcn_hq;
   pthread_mutex_lock(&hq->lock);
   list_for_each_entry_safe(struct bcn_hq_job, j, &hq->jobs, link) {
      if (j->wcb == wcb) {
         j->wcb = NULL;
         j->queue = queue;
      } else if (j->wi && j->wi->bcn_touch == wcb && j->queue && j->queue != queue) {
         hq_drop_logged(device, hq, j, "image used on another queue");
      }
   }
   pthread_mutex_unlock(&hq->lock);
}

static bool
hq_layout_ok(VkImageLayout l)
{
   return l == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || l == VK_IMAGE_LAYOUT_GENERAL ||
          l == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL || l == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ||
          l == VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
}

static void
hq_reap(struct wrapper_device *device, struct bcn_hq *hq, bool wait)
{
   VkDevice d = device->dispatch_handle;
   list_for_each_entry_safe(struct bcn_hq_batch, b, &hq->batches, link) {
      if (wait)
         device->dispatch_table.WaitForFences(d, 1, &b->fence, VK_TRUE, UINT64_MAX);
      else if (device->dispatch_table.GetFenceStatus(d, b->fence) != VK_SUCCESS)
         continue;
      device->dispatch_table.DestroyFence(d, b->fence, NULL);
      device->dispatch_table.FreeCommandBuffers(d, b->pool, 1, &b->cb);
      device->dispatch_table.DestroyBuffer(d, b->buffer, NULL);
      device->dispatch_table.FreeMemory(d, b->memory, NULL);
      list_del(&b->link);
      free(b);
   }
}

static uint32_t
hq_host_memory_type(struct wrapper_device *device, uint32_t bits)
{
   const VkPhysicalDeviceMemoryProperties *p = &device->physical->memory_properties;
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < p->memoryTypeCount; i++)
      if ((bits & (1u << i)) && (p->memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

/* Called inside vkQueueSubmit before the application's batches: copies the
 * finished jobs whose upload was submitted to this queue earlier. */
void
wrapper_bcn_hq_flush(struct wrapper_queue *queue)
{
   struct wrapper_device *device = queue->device;
   if (!__atomic_load_n(&device->bcn_hq_active, __ATOMIC_RELAXED) || !device->bcn_hq)
      return;
   struct bcn_hq *hq = device->bcn_hq;
   VkDevice d = device->dispatch_handle;
   uint32_t family = queue->vk.queue_family_index;
   pthread_mutex_lock(&hq->lock);
   hq_reap(device, hq, false);

   size_t total = 0;
   uint32_t count = 0;
   list_for_each_entry(struct bcn_hq_job, j, &hq->jobs, link) {
      if (j->state == HQ_READY && j->queue == queue && j->wi && hq_layout_ok(j->wi->bcn_layout)) {
         total += j->layer_out * j->sub.layerCount;
         count++;
      }
   }
   if (!count || family >= ARRAY_SIZE(hq->pools)) {
      pthread_mutex_unlock(&hq->lock);
      return;
   }

   struct bcn_hq_batch *b = calloc(1, sizeof(*b));
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = total,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = family,
   };
   void *map = NULL;
   bool ok = b && device->dispatch_table.CreateBuffer(d, &bci, NULL, &b->buffer) == VK_SUCCESS;
   if (ok) {
      VkMemoryRequirements mr;
      device->dispatch_table.GetBufferMemoryRequirements(d, b->buffer, &mr);
      VkMemoryAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size,
         .memoryTypeIndex = hq_host_memory_type(device, mr.memoryTypeBits),
      };
      ok = ai.memoryTypeIndex != UINT32_MAX &&
           device->dispatch_table.AllocateMemory(d, &ai, NULL, &b->memory) == VK_SUCCESS &&
           device->dispatch_table.BindBufferMemory(d, b->buffer, b->memory, 0) == VK_SUCCESS &&
           device->dispatch_table.MapMemory(d, b->memory, 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS;
   }
   if (ok && !hq->pools[family])
      ok = device->dispatch_table.CreateCommandPool(d, &cpci, NULL, &hq->pools[family]) ==
           VK_SUCCESS;
   if (ok) {
      b->pool = hq->pools[family];
      VkCommandBufferAllocateInfo cbai = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = b->pool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      ok = device->dispatch_table.AllocateCommandBuffers(d, &cbai, &b->cb) == VK_SUCCESS &&
           device->dispatch_table.CreateFence(d, &fci, NULL, &b->fence) == VK_SUCCESS;
   }
   if (!ok) {
      if (b) {
         if (b->cb)
            device->dispatch_table.FreeCommandBuffers(d, b->pool, 1, &b->cb);
         if (b->buffer)
            device->dispatch_table.DestroyBuffer(d, b->buffer, NULL);
         if (b->memory)
            device->dispatch_table.FreeMemory(d, b->memory, NULL);
         free(b);
      }
      pthread_mutex_unlock(&hq->lock);
      return;
   }
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   device->dispatch_table.BeginCommandBuffer(b->cb, &bi);
   VkDeviceSize at = 0;
   double now = now_ms();
   list_for_each_entry_safe(struct bcn_hq_job, j, &hq->jobs, link) {
      if (!(j->state == HQ_READY && j->queue == queue && j->wi && hq_layout_ok(j->wi->bcn_layout)))
         continue;
      VkImageLayout old = j->wi->bcn_layout;
      VkImageLayout dst = old == VK_IMAGE_LAYOUT_GENERAL ? old : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      VkImageMemoryBarrier to = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = old,
         .newLayout = dst,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = j->wi->dispatch_handle,
         .subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, j->sub.mipLevel, 1, j->sub.baseArrayLayer,
            j->sub.layerCount,
         },
      };
      device->dispatch_table.CmdPipelineBarrier(b->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to);
      size_t len = j->layer_out * j->sub.layerCount;
      memcpy((uint8_t *)map + at, j->out, len);
      VkBufferImageCopy copy = {
         .bufferOffset = at,
         .bufferRowLength = (j->extent.width + j->block - 1) / j->block * j->block,
         .bufferImageHeight = (j->extent.height + j->block - 1) / j->block * j->block,
         .imageSubresource = j->sub,
         .imageOffset = j->offset,
         .imageExtent = j->extent,
      };
      device->dispatch_table.CmdCopyBufferToImage(b->cb, b->buffer, j->wi->dispatch_handle, dst,
                                                  1, &copy);
      VkImageMemoryBarrier back = to;
      back.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      back.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      back.oldLayout = dst;
      back.newLayout = old;
      device->dispatch_table.CmdPipelineBarrier(b->cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &back);
      at += len;
      if ((uint64_t)j->extent.width * j->extent.height >= 1024 * 1024)
         wrapper_diag_append("[BCN] hq %s %ux%u -> astc %ux%u: final copied %.0f ms after "
                             "the upload (astcenc %.0f ms, %u threads)\n", hq_format_name(j->format),
                             j->extent.width, j->extent.height, j->block, j->block,
                             now - j->t_capture, j->encode_ms, hq->threads);
      hq_free_job(hq, j);
      __atomic_sub_fetch(&device->bcn_hq_active, 1, __ATOMIC_RELAXED);
   }
   device->dispatch_table.EndCommandBuffer(b->cb);
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &b->cb,
   };
   if (device->dispatch_table.QueueSubmit(queue->dispatch_handle, 1, &si, b->fence) == VK_SUCCESS) {
      list_addtail(&b->link, &hq->batches);
   } else {
      device->dispatch_table.DestroyFence(d, b->fence, NULL);
      device->dispatch_table.FreeCommandBuffers(d, b->pool, 1, &b->cb);
      device->dispatch_table.DestroyBuffer(d, b->buffer, NULL);
      device->dispatch_table.FreeMemory(d, b->memory, NULL);
      free(b);
   }
   pthread_mutex_unlock(&hq->lock);
}

void
wrapper_bcn_hq_finish(struct wrapper_device *device)
{
   struct bcn_hq *hq = device->bcn_hq;
   if (!hq)
      return;
   pthread_mutex_lock(&hq->lock);
   hq->stop = true;
   pthread_cond_signal(&hq->wake);
   pthread_mutex_unlock(&hq->lock);
   pthread_join(hq->thread, NULL);
   hq_reap(device, hq, true);
   list_for_each_entry_safe(struct bcn_hq_job, j, &hq->jobs, link) {
      j->wi = NULL;
      hq_free_job(hq, j);
   }
   for (uint32_t i = 0; i < ARRAY_SIZE(hq->pools); i++)
      if (hq->pools[i])
         device->dispatch_table.DestroyCommandPool(device->dispatch_handle, hq->pools[i], NULL);
   pthread_cond_destroy(&hq->wake);
   pthread_mutex_destroy(&hq->lock);
   free(hq);
   device->bcn_hq = NULL;
}
