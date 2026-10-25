#include <stdlib.h>
#include <string.h>

#include "wrapper_bcn_gpu.h"
#include "wrapper_log.h"
#include "util/macros.h"

/* An emulated BC image keeps the blocks it was given in a device buffer, the
 * shadow, packed per level and slice. Readbacks are served from it, so they
 * return the application's BC data rather than the stored format. */

static uint32_t
bcn_block_bytes(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
   case VK_FORMAT_BC4_UNORM_BLOCK:
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return 8;
   default:
      return 16;
   }
}

static int
bcn_shadow_enabled(void)
{
   static int on = -1;
   if (on == -1)
      on = getenv("WRAPPER_BCN_SHADOW") ? atoi(getenv("WRAPPER_BCN_SHADOW")) : 1;
   return on;
}

static uint32_t
bcn_memory_type(struct wrapper_device *device, uint32_t bits, VkMemoryPropertyFlags want)
{
   const VkPhysicalDeviceMemoryProperties *props = &device->physical->memory_properties;
   for (uint32_t i = 0; i < props->memoryTypeCount; i++)
      if ((bits & (1u << i)) && (props->memoryTypes[i].propertyFlags & want) == want)
         return i;
   for (uint32_t i = 0; i < props->memoryTypeCount; i++)
      if (bits & (1u << i))
         return i;
   return UINT32_MAX;
}

static bool
bcn_alloc_buffer(struct wrapper_device *device, VkDeviceSize size, VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags props, VkBuffer *buffer, VkDeviceMemory *memory)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &bci, NULL,
                                           buffer) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(device->dispatch_handle, *buffer, &mr);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = bcn_memory_type(device, mr.memoryTypeBits, props),
   };
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(device->dispatch_handle, &ai, NULL,
                                             memory) != VK_SUCCESS) {
      device->dispatch_table.DestroyBuffer(device->dispatch_handle, *buffer, NULL);
      return false;
   }
   device->dispatch_table.BindBufferMemory(device->dispatch_handle, *buffer, *memory, 0);
   return true;
}

static uint32_t
mip_extent(uint32_t v, uint32_t level)
{
   return MAX2(1u, v >> level);
}

struct wrapper_bcn_shadow *
wrapper_bcn_shadow_get(struct wrapper_device *device, struct wrapper_image *wi)
{
   if (!bcn_shadow_enabled() || wi->mip_drop)
      return NULL;
   simple_mtx_lock(&device->bcn_gpu_mutex);
   struct wrapper_bcn_shadow *s = wi->bcn_shadow;
   if (s || wi->bcn_shadow_failed) {
      simple_mtx_unlock(&device->bcn_gpu_mutex);
      return s;
   }
   s = calloc(1, sizeof(*s));
   if (!s) {
      simple_mtx_unlock(&device->bcn_gpu_mutex);
      return NULL;
   }
   s->block_bytes = bcn_block_bytes(wi->info.format);
   s->levels = MIN2(wi->info.mipLevels, ARRAY_SIZE(s->level_offset));
   s->layers = wi->info.arrayLayers;
   VkDeviceSize size = 0;
   for (uint32_t m = 0; m < s->levels; m++) {
      s->level_offset[m] = size;
      s->blocks_x[m] = (mip_extent(wi->info.extent.width, m) + 3) / 4;
      s->blocks_y[m] = (mip_extent(wi->info.extent.height, m) + 3) / 4;
      s->depth[m] = mip_extent(wi->info.extent.depth, m);
      size += (VkDeviceSize)s->blocks_x[m] * s->blocks_y[m] * s->block_bytes *
              s->depth[m] * s->layers;
   }
   s->size = size;
   if (!bcn_alloc_buffer(device, size,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &s->buffer, &s->memory)) {
      WRAPPER_LOG(error, "BCn: no shadow buffer for a %ux%u image",
                  wi->info.extent.width, wi->info.extent.height);
      free(s);
      wi->bcn_shadow_failed = true;
      simple_mtx_unlock(&device->bcn_gpu_mutex);
      return NULL;
   }
   wi->bcn_shadow = s;
   simple_mtx_unlock(&device->bcn_gpu_mutex);
   return s;
}

void
wrapper_bcn_shadow_free(struct wrapper_device *device, struct wrapper_image *wi)
{
   struct wrapper_bcn_shadow *s = wi->bcn_shadow;
   if (!s)
      return;
   device->dispatch_table.DestroyBuffer(device->dispatch_handle, s->buffer, NULL);
   device->dispatch_table.FreeMemory(device->dispatch_handle, s->memory, NULL);
   free(s);
   wi->bcn_shadow = NULL;
}

static VkDeviceSize
shadow_offset(const struct wrapper_bcn_shadow *s, uint32_t level, uint32_t slice,
              uint32_t block_row, uint32_t block_col)
{
   return s->level_offset[level] +
          (((VkDeviceSize)slice * s->blocks_y[level] + block_row) * s->blocks_x[level] +
           block_col) * s->block_bytes;
}

static void
bcn_barrier(struct wrapper_command_buffer *wcb)
{
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
   };
   wcb->device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &mb, 0, NULL, 0, NULL);
}

/* A copy region in blocks: level, first slice (layer * depth + z), slice
 * count, and the block rectangle. */
struct bcn_block_region {
   uint32_t level, slice, slices;
   uint32_t bx, by, nbx, nby;
};

static bool
bcn_region_from_copy(const struct wrapper_bcn_shadow *s, const VkImageSubresourceLayers *sub,
                     VkOffset3D off, VkExtent3D ext, struct bcn_block_region *r)
{
   if (sub->mipLevel >= s->levels)
      return false;
   uint32_t m = sub->mipLevel;
   uint32_t layers = sub->layerCount == VK_REMAINING_ARRAY_LAYERS ?
      s->layers - sub->baseArrayLayer : sub->layerCount;
   r->level = m;
   if (s->depth[m] > 1) {
      r->slice = off.z;
      r->slices = MAX2(ext.depth, 1);
   } else {
      r->slice = sub->baseArrayLayer;
      r->slices = layers;
   }
   r->bx = off.x / 4;
   r->by = off.y / 4;
   r->nbx = (ext.width + 3) / 4;
   r->nby = (ext.height + 3) / 4;
   return r->bx + r->nbx <= s->blocks_x[m] && r->by + r->nby <= s->blocks_y[m] &&
          r->slice + r->slices <= s->depth[m] * s->layers;
}

/* Copies between a BC buffer region and the shadow, in either direction. */
static void
bcn_shadow_buffer_copy(struct wrapper_command_buffer *wcb, const struct wrapper_bcn_shadow *s,
                       VkBuffer other, VkDeviceSize base, VkDeviceSize row_pitch,
                       VkDeviceSize slice_pitch, const struct bcn_block_region *r,
                       bool to_shadow)
{
   struct wrapper_device *device = wcb->device;
   VkDeviceSize row_bytes = (VkDeviceSize)r->nbx * s->block_bytes;
   bool whole_rows = r->nbx == s->blocks_x[r->level] && row_pitch == row_bytes;
   uint32_t rows = whole_rows ? 1 : r->nby;
   uint32_t count = r->slices * rows;
   VkBufferCopy stack[64];
   VkBufferCopy *copies = count <= ARRAY_SIZE(stack) ? stack : malloc(count * sizeof(*copies));
   if (!copies)
      return;
   uint32_t n = 0;
   for (uint32_t z = 0; z < r->slices; z++) {
      for (uint32_t y = 0; y < rows; y++) {
         VkDeviceSize o = base + z * slice_pitch + y * row_pitch;
         VkDeviceSize sh = shadow_offset(s, r->level, r->slice + z, r->by + y, r->bx);
         VkDeviceSize size = whole_rows ? row_bytes * r->nby : row_bytes;
         copies[n++] = to_shadow ? (VkBufferCopy){ o, sh, size } : (VkBufferCopy){ sh, o, size };
      }
   }
   if (to_shadow)
      device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, other, s->buffer, n, copies);
   else
      device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, s->buffer, other, n, copies);
   if (copies != stack)
      free(copies);
}

static void
bcn_buffer_pitches(const struct wrapper_bcn_shadow *s, const VkBufferImageCopy *c,
                   VkDeviceSize *row, VkDeviceSize *slice)
{
   uint32_t w = c->bufferRowLength ? c->bufferRowLength : c->imageExtent.width;
   uint32_t h = c->bufferImageHeight ? c->bufferImageHeight : c->imageExtent.height;
   *row = (VkDeviceSize)((w + 3) / 4) * s->block_bytes;
   *slice = *row * ((h + 3) / 4);
}

static void
bcn_shadow_regions(struct wrapper_command_buffer *wcb, struct wrapper_bcn_shadow *s,
                   VkBuffer buffer, uint32_t count, const VkBufferImageCopy *regions,
                   bool to_shadow)
{
   bcn_barrier(wcb);
   for (uint32_t i = 0; i < count; i++) {
      const VkBufferImageCopy *c = &regions[i];
      struct bcn_block_region r;
      if (!bcn_region_from_copy(s, &c->imageSubresource, c->imageOffset, c->imageExtent, &r))
         continue;
      VkDeviceSize row, slice;
      bcn_buffer_pitches(s, c, &row, &slice);
      bcn_shadow_buffer_copy(wcb, s, buffer, c->bufferOffset, row, slice, &r, to_shadow);
   }
}

void
wrapper_bcn_shadow_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                          VkBuffer src, uint32_t count, const VkBufferImageCopy *regions)
{
   struct wrapper_bcn_shadow *s = wrapper_bcn_shadow_get(wcb->device, wi);
   if (s)
      bcn_shadow_regions(wcb, s, src, count, regions, true);
}

bool
wrapper_bcn_shadow_readback(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                            VkBuffer dst, uint32_t count, const VkBufferImageCopy *regions)
{
   struct wrapper_bcn_shadow *s = wrapper_bcn_shadow_get(wcb->device, wi);
   if (!s)
      return false;
   bcn_shadow_regions(wcb, s, dst, count, regions, false);
   return true;
}
