#include <stdlib.h>
#include <string.h>

#include "wrapper_bcn_gpu.h"
#include "wrapper_bcdec.h"
#include "wrapper_bcn_decode_spv.h"
#include "wrapper_log.h"
#include "util/macros.h"

/* An emulated BC image keeps the blocks it was given in a device buffer, the
 * shadow, packed per level and slice. Readbacks are served from it, so they
 * return the application's BC data rather than the stored format, and the GPU
 * decoder reads it to produce the stored texels. */

struct wrapper_bcn_decode_pc {
   uint32_t src[2];
   uint32_t dst[2];
   uint32_t src_row_pitch;
   uint32_t src_slice_pitch;
   uint32_t dst_row_pitch;
   uint32_t dst_slice_pitch;
   uint32_t blocks_x;
   uint32_t blocks_y;
   uint32_t format;
};

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
bcn_decode_format_id(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return 0;
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return 1;
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK: return 2;
   case VK_FORMAT_BC3_UNORM_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK: return 3;
   case VK_FORMAT_BC4_UNORM_BLOCK: return 4;
   case VK_FORMAT_BC4_SNORM_BLOCK: return 5;
   case VK_FORMAT_BC5_UNORM_BLOCK: return 6;
   case VK_FORMAT_BC5_SNORM_BLOCK: return 7;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK: return 8;
   case VK_FORMAT_BC6H_SFLOAT_BLOCK: return 9;
   case VK_FORMAT_BC7_UNORM_BLOCK:
   case VK_FORMAT_BC7_SRGB_BLOCK: return 10;
   default: return -1;
   }
}

static uint32_t
bcn_decoded_texel_size(VkFormat bcn_format)
{
   switch (get_decode_format_for_bcn(bcn_format)) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM: return 1;
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM: return 2;
   case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
   default: return 4;
   }
}

int
wrapper_bcn_gpu_requested(void)
{
   static int on = -1;
   if (on == -1)
      on = getenv("WRAPPER_BCN_GPU") ? atoi(getenv("WRAPPER_BCN_GPU")) : 0;
   return on;
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
                 VkMemoryPropertyFlags props, VkBuffer *buffer, VkDeviceMemory *memory,
                 VkDeviceAddress *address)
{
   bool bda = device->bcn_bda;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage | (bda ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &bci, NULL,
                                           buffer) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(device->dispatch_handle, *buffer, &mr);
   VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = bda ? &flags : NULL,
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
   *address = 0;
   if (bda) {
      VkBufferDeviceAddressInfo dai = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
         .buffer = *buffer,
      };
      *address = device->dispatch_table.GetBufferDeviceAddress(device->dispatch_handle, &dai);
   }
   return true;
}

/* A device-local buffer freed with the command buffer. */
static struct wrapper_buffer *
bcn_transient_buffer(struct wrapper_command_buffer *wcb, VkDeviceSize size,
                     VkBufferUsageFlags usage, VkDeviceAddress *address)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_buffer *b = vk_object_zalloc(&device->vk, &device->vk.alloc,
      sizeof(struct wrapper_buffer), VK_OBJECT_TYPE_BUFFER);
   if (!b)
      return NULL;
   if (!bcn_alloc_buffer(device, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &b->dispatch_handle, &b->memory, address)) {
      vk_object_free(&device->vk, &device->vk.alloc, b);
      return NULL;
   }
   b->device = device;
   b->wcb = wcb;
   b->size = size;
   list_addtail(&b->link, &wcb->bcn_transients);
   return b;
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
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &s->buffer, &s->memory,
                         &s->address)) {
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

static void
bcn_gpu_init(struct wrapper_device *device)
{
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(wrapper_bcn_decode_spv),
      .pCode = wrapper_bcn_decode_spv,
   };
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct wrapper_bcn_decode_pc),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   device->bcn_gpu_state = -1;
   if (!device->bcn_bda ||
       device->dispatch_table.CreateShaderModule(device->dispatch_handle, &smci, NULL,
                                                 &device->bcn_shader) != VK_SUCCESS ||
       device->dispatch_table.CreatePipelineLayout(device->dispatch_handle, &plci, NULL,
                                                   &device->bcn_pipe_layout) != VK_SUCCESS)
      return;
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = device->bcn_shader,
         .pName = "main",
      },
      .layout = device->bcn_pipe_layout,
   };
   if (device->dispatch_table.CreateComputePipelines(device->dispatch_handle, VK_NULL_HANDLE,
                                                     1, &cpci, NULL,
                                                     &device->bcn_pipeline) != VK_SUCCESS)
      return;
   device->bcn_gpu_state = 1;
}

static bool
bcn_gpu_ready(struct wrapper_device *device)
{
   simple_mtx_lock(&device->bcn_gpu_mutex);
   if (device->bcn_gpu_state == 0) {
      bcn_gpu_init(device);
      const char *state = device->bcn_gpu_state == 1 ? "ready" :
         device->bcn_bda ? "failed to initialise" : "needs bufferDeviceAddress";
      WRAPPER_LOG(info, "BCn GPU decode %s", state);
      wrapper_diag_append("[BCN] GPU decode %s\n", state);
   }
   bool ready = device->bcn_gpu_state == 1;
   simple_mtx_unlock(&device->bcn_gpu_mutex);
   return ready;
}

void
wrapper_bcn_gpu_finish(struct wrapper_device *device)
{
   if (device->bcn_pipeline)
      device->dispatch_table.DestroyPipeline(device->dispatch_handle, device->bcn_pipeline, NULL);
   if (device->bcn_pipe_layout)
      device->dispatch_table.DestroyPipelineLayout(device->dispatch_handle,
                                                   device->bcn_pipe_layout, NULL);
   if (device->bcn_shader)
      device->dispatch_table.DestroyShaderModule(device->dispatch_handle, device->bcn_shader,
                                                 NULL);
}

void
wrapper_cmd_track_pipeline(struct wrapper_command_buffer *wcb, VkPipelineBindPoint bind_point,
                           VkPipeline pipeline)
{
   if (bind_point == VK_PIPELINE_BIND_POINT_COMPUTE)
      wcb->compute_pipeline = pipeline;
}

void
wrapper_cmd_track_push(struct wrapper_command_buffer *wcb, VkPipelineLayout layout,
                       VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                       const void *data)
{
   if (size > sizeof(wcb->push[0].data))
      return;
   uint32_t i;
   for (i = 0; i < wcb->push_count; i++) {
      const struct wrapper_push_record *p = &wcb->push[i];
      if (p->layout == layout && p->stages == stages && p->offset == offset &&
          p->size == size)
         break;
   }
   if (i == wcb->push_count && wcb->push_count < ARRAY_SIZE(wcb->push))
      wcb->push_count++;
   else if (i == wcb->push_count)
      i = 0;
   memmove(&wcb->push[i], &wcb->push[i + 1],
           (wcb->push_count - 1 - i) * sizeof(wcb->push[0]));
   struct wrapper_push_record *p = &wcb->push[wcb->push_count - 1];
   p->layout = layout;
   p->stages = stages;
   p->offset = offset;
   p->size = size;
   memcpy(p->data, data, size);
}

void
wrapper_cmd_track_reset(struct wrapper_command_buffer *wcb)
{
   wcb->compute_pipeline = VK_NULL_HANDLE;
   wcb->push_count = 0;
}

/* The decode binds its own pipeline and push constants in the application's
 * command buffer; put back what the application had set. */
static void
bcn_restore_compute(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   if (wcb->compute_pipeline)
      device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
         VK_PIPELINE_BIND_POINT_COMPUTE, wcb->compute_pipeline);
   for (uint32_t i = 0; i < wcb->push_count; i++)
      device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, wcb->push[i].layout,
         wcb->push[i].stages, wcb->push[i].offset, wcb->push[i].size, wcb->push[i].data);
}

static bool
bcn_gpu_can_decode(const struct wrapper_image *wi)
{
   return !wi->mip_drop && bcn_decode_format_id(wi->info.format) >= 0 &&
          get_format_for_bcn(wi->info.format) == get_decode_format_for_bcn(wi->info.format);
}

/* Decodes a block region of the shadow into the image, which is in a layout
 * for transfer writes. */
static bool
bcn_gpu_decode(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
               VkImageLayout layout, const VkImageSubresourceLayers *sub,
               VkOffset3D off, VkExtent3D ext, const struct bcn_block_region *r)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_bcn_shadow *s = wi->bcn_shadow;
   if (!s || !bcn_gpu_can_decode(wi) || !bcn_gpu_ready(device))
      return false;
   uint32_t ts = bcn_decoded_texel_size(wi->info.format);
   uint32_t row = r->nbx * 4 * ts, slice = row * r->nby * 4;
   VkDeviceAddress dst;
   struct wrapper_buffer *tmp = bcn_transient_buffer(wcb, (VkDeviceSize)slice * r->slices,
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &dst);
   if (!tmp)
      return false;

   VkDeviceAddress src = s->address + shadow_offset(s, r->level, r->slice, r->by, r->bx);
   struct wrapper_bcn_decode_pc pc = {
      .src = { (uint32_t)src, (uint32_t)(src >> 32) },
      .dst = { (uint32_t)dst, (uint32_t)(dst >> 32) },
      .src_row_pitch = s->blocks_x[r->level] * s->block_bytes,
      .src_slice_pitch = s->blocks_x[r->level] * s->blocks_y[r->level] * s->block_bytes,
      .dst_row_pitch = row,
      .dst_slice_pitch = slice,
      .blocks_x = r->nbx,
      .blocks_y = r->nby,
      .format = bcn_decode_format_id(wi->info.format),
   };
   bcn_barrier(wcb);
   device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
      VK_PIPELINE_BIND_POINT_COMPUTE, device->bcn_pipeline);
   device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, device->bcn_pipe_layout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   device->dispatch_table.CmdDispatch(wcb->dispatch_handle, (r->nbx + 7) / 8,
      (r->nby + 7) / 8, r->slices);
   bcn_barrier(wcb);
   bcn_restore_compute(wcb);

   VkBufferImageCopy copy = {
      .bufferRowLength = r->nbx * 4,
      .bufferImageHeight = r->nby * 4,
      .imageSubresource = *sub,
      .imageOffset = off,
      .imageExtent = ext,
   };
   device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle, tmp->dispatch_handle,
      wi->dispatch_handle, layout, 1, &copy);
   return true;
}

/* GPU upload: the blocks are already in the shadow; decode them into the
 * image. Returns how many leading regions it recorded; the CPU path takes the
 * rest. */
uint32_t
wrapper_bcn_gpu_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       VkImageLayout layout, uint32_t count, const VkBufferImageCopy *regions)
{
   struct wrapper_bcn_shadow *s = wi->bcn_shadow;
   uint32_t done = 0;
   if (!s || !bcn_gpu_can_decode(wi) || !bcn_gpu_ready(wcb->device))
      return 0;
   for (; done < count; done++) {
      const VkBufferImageCopy *c = &regions[done];
      struct bcn_block_region r;
      if (!bcn_region_from_copy(s, &c->imageSubresource, c->imageOffset, c->imageExtent, &r) ||
          !bcn_gpu_decode(wcb, wi, layout, &c->imageSubresource, c->imageOffset,
                          c->imageExtent, &r))
         break;
   }
   return done;
}

/* One vkCmdCopyImage region with an emulated BC image on either side or both.
 * The stored texels of an emulated image are not the BC blocks the copy is
 * defined on, so the blocks move through the shadows: a BC source is read from
 * its shadow, and a BC destination gets the blocks in its shadow and is decoded
 * from there. Returns false when the region should go to the driver as is. */
bool
wrapper_bcn_copy_image_region(struct wrapper_command_buffer *wcb,
                              struct wrapper_image *src, bool src_emulated,
                              VkImageLayout src_layout,
                              struct wrapper_image *dst, bool dst_emulated,
                              VkImageLayout dst_layout, const VkImageCopy *c)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_bcn_shadow *ss = src_emulated ? wrapper_bcn_shadow_get(device, src) : NULL;
   struct wrapper_bcn_shadow *ds = dst_emulated ? wrapper_bcn_shadow_get(device, dst) : NULL;
   struct bcn_block_region sr, dr;

   if ((src_emulated && !ss) || (dst_emulated && !ds))
      return false;
   if (ss && !bcn_region_from_copy(ss, &c->srcSubresource, c->srcOffset, c->extent, &sr))
      return false;

   VkExtent3D dst_extent = c->extent;
   if (ds) {
      if (!ss) {
         uint32_t m = c->dstSubresource.mipLevel;
         dst_extent.width = MIN2(c->extent.width * 4,
            mip_extent(dst->info.extent.width, m) - (uint32_t)c->dstOffset.x);
         dst_extent.height = MIN2(c->extent.height * 4,
            mip_extent(dst->info.extent.height, m) - (uint32_t)c->dstOffset.y);
      }
      if (!bcn_region_from_copy(ds, &c->dstSubresource, c->dstOffset, dst_extent, &dr))
         return false;
   }

   bcn_barrier(wcb);

   if (ss && ds) {
      if (sr.nbx != dr.nbx || sr.nby != dr.nby || sr.slices != dr.slices)
         return false;
      for (uint32_t z = 0; z < sr.slices; z++) {
         VkBufferCopy bc[64];
         uint32_t n = 0;
         for (uint32_t y = 0; y < sr.nby; y++) {
            bc[n++] = (VkBufferCopy){
               shadow_offset(ss, sr.level, sr.slice + z, sr.by + y, sr.bx),
               shadow_offset(ds, dr.level, dr.slice + z, dr.by + y, dr.bx),
               (VkDeviceSize)sr.nbx * ss->block_bytes,
            };
            if (n == ARRAY_SIZE(bc) || y + 1 == sr.nby) {
               device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, ss->buffer,
                                                    ds->buffer, n, bc);
               n = 0;
            }
         }
      }
      if (bcn_decode_format_id(src->info.format) == bcn_decode_format_id(dst->info.format) &&
          get_format_for_bcn(src->info.format) == get_format_for_bcn(dst->info.format)) {
         device->dispatch_table.CmdCopyImage(wcb->dispatch_handle, src->dispatch_handle,
            src_layout, dst->dispatch_handle, dst_layout, 1, c);
         return true;
      }
   } else if (ss) {
      VkBufferImageCopy bic = {
         .bufferOffset = shadow_offset(ss, sr.level, sr.slice, sr.by, sr.bx),
         .bufferRowLength = ss->blocks_x[sr.level],
         .bufferImageHeight = ss->blocks_y[sr.level],
         .imageSubresource = c->dstSubresource,
         .imageOffset = c->dstOffset,
         .imageExtent = { sr.nbx, sr.nby, ss->depth[sr.level] > 1 ? sr.slices : 1 },
      };
      device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle, ss->buffer,
         dst->dispatch_handle, dst_layout, 1, &bic);
      return true;
   } else {
      VkBufferImageCopy bic = {
         .bufferOffset = shadow_offset(ds, dr.level, dr.slice, dr.by, dr.bx),
         .bufferRowLength = ds->blocks_x[dr.level],
         .bufferImageHeight = ds->blocks_y[dr.level],
         .imageSubresource = c->srcSubresource,
         .imageOffset = c->srcOffset,
         .imageExtent = { dr.nbx, dr.nby, ds->depth[dr.level] > 1 ? dr.slices : 1 },
      };
      device->dispatch_table.CmdCopyImageToBuffer(wcb->dispatch_handle, src->dispatch_handle,
         src_layout, ds->buffer, 1, &bic);
   }

   if (!bcn_gpu_decode(wcb, dst, dst_layout, &c->dstSubresource, c->dstOffset, dst_extent,
                       &dr)) {
      static bool logged;
      if (!logged)
         WRAPPER_LOG(error, "BCn: no GPU decode for a copy into format %d stored as %d",
                     dst->info.format, get_format_for_bcn(dst->info.format));
      logged = true;
   }
   return true;
}
