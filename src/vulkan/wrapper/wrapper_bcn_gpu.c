#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

#include "wrapper_bcn_gpu.h"
#include "wrapper_bcdec.h"
#include "wrapper_astcenc.h"
#include "wrapper_bcn_decode_spv.h"
#include "wrapper_bcn_encode_spv.h"
#include "wrapper_astc_encode_spv.h"
#include "wrapper_astc_hdr_encode_spv.h"
#include "wrapper_astc_tables.h"
#include "wrapper_astc_fast_spv.h"
#include "wrapper_astc_fast_sg_spv.h"
#include "wrapper_log.h"
#include "util/macros.h"

/* GPU work for emulated BC images. A readback, or a copy to or from another
 * image, needs BC blocks where the image holds ASTC or decoded texels: the
 * stored texels are blitted to an RGBA image (the driver decodes ASTC) and
 * BC-encoded by a compute shader; blocks copied in are decoded by another.
 * Uploads are decoded and encoded to ASTC on the GPU once the shaders they
 * need are built; the CPU path serves them until then, and when forced with
 * WRAPPER_BCN_FORCE_CPU=1. Nothing is kept beyond the
 * command buffer: the BC blocks of the original upload are not stored. */

int
wrapper_bcn_gpu_path(void)
{
   static int on = -1;
   if (on == -1) {
      const char *cpu = getenv("WRAPPER_BCN_FORCE_CPU");
      on = !(cpu && atoi(cpu));
   }
   return on;
}

static uint32_t
bcn_astc_block(VkFormat stored);

struct bcn_pc {
   uint32_t src[2];
   uint32_t dst[2];
   uint32_t src_row_pitch;
   uint32_t src_slice_pitch;
   uint32_t dst_row_pitch;
   uint32_t dst_slice_pitch;
   uint32_t blocks_x;
   uint32_t blocks_y;
   uint32_t format;
   uint32_t flags;
   uint32_t width;
   uint32_t height;
   uint32_t tables[2];
   uint32_t bc[2];
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
bcn_format_id(VkFormat format)
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

/* The RGBA format the stored texels are blitted to before encoding. */
static VkFormat
bcn_unpacked_format(VkFormat bcn_format)
{
   switch (bcn_format) {
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
   case VK_FORMAT_BC4_SNORM_BLOCK:
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return VK_FORMAT_R8G8B8A8_SNORM;
   default:
      return get_decode_format_for_bcn(bcn_format) == VK_FORMAT_R8G8B8A8_SRGB ?
         VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
   }
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

static VkDeviceMemory
bcn_alloc_memory(struct wrapper_device *device, const VkMemoryRequirements *mr, bool address)
{
   VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = address ? &flags : NULL,
      .allocationSize = mr->size,
      .memoryTypeIndex = bcn_memory_type(device, mr->memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory memory = VK_NULL_HANDLE;
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(device->dispatch_handle, &ai, NULL,
                                             &memory) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return memory;
}

static struct wrapper_buffer *
bcn_transient(struct wrapper_command_buffer *wcb)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_buffer *b = vk_object_zalloc(&device->vk, &device->vk.alloc,
      sizeof(struct wrapper_buffer), VK_OBJECT_TYPE_BUFFER);
   if (!b)
      return NULL;
   b->device = device;
   b->wcb = wcb;
   list_addtail(&b->link, &wcb->bcn_transients);
   return b;
}

/* A device-local buffer with a device address, freed with the command buffer. */
static struct wrapper_buffer *
bcn_transient_buffer(struct wrapper_command_buffer *wcb, VkDeviceSize size,
                     VkDeviceAddress *address)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_buffer *b = bcn_transient(wcb);
   if (!b)
      return NULL;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &bci, NULL,
                                           &b->dispatch_handle) != VK_SUCCESS)
      return NULL;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(device->dispatch_handle,
                                                      b->dispatch_handle, &mr);
   b->memory = bcn_alloc_memory(device, &mr, true);
   if (!b->memory)
      return NULL;
   device->dispatch_table.BindBufferMemory(device->dispatch_handle, b->dispatch_handle,
                                           b->memory, 0);
   VkBufferDeviceAddressInfo dai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = b->dispatch_handle,
   };
   *address = device->dispatch_table.GetBufferDeviceAddress(device->dispatch_handle, &dai);
   b->size = size;
   return b;
}

/* A mapped host-visible buffer, freed with the command buffer. */
static struct wrapper_buffer *
bcn_transient_host_buffer(struct wrapper_command_buffer *wcb, VkDeviceSize size, void **map)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_buffer *b = bcn_transient(wcb);
   if (!b)
      return NULL;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &bci, NULL,
                                           &b->dispatch_handle) != VK_SUCCESS)
      return NULL;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(device->dispatch_handle,
                                                      b->dispatch_handle, &mr);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = bcn_memory_type(device, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(device->dispatch_handle, &ai, NULL,
                                             &b->memory) != VK_SUCCESS ||
       device->dispatch_table.BindBufferMemory(device->dispatch_handle, b->dispatch_handle,
                                               b->memory, 0) != VK_SUCCESS ||
       device->dispatch_table.MapMemory(device->dispatch_handle, b->memory, 0, VK_WHOLE_SIZE,
                                        0, map) != VK_SUCCESS)
      return NULL;
   b->size = size;
   return b;
}

/* A 2D image (array) for unpacked texels, freed with the command buffer. */
static VkImage
bcn_transient_image(struct wrapper_command_buffer *wcb, VkFormat format, uint32_t w,
                    uint32_t h, uint32_t layers)
{
   struct wrapper_device *device = wcb->device;
   struct wrapper_buffer *b = bcn_transient(wcb);
   if (!b)
      return VK_NULL_HANDLE;
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = { w, h, 1 },
      .mipLevels = 1,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   if (device->dispatch_table.CreateImage(device->dispatch_handle, &ici, NULL,
                                          &b->bcn_image) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   VkMemoryRequirements mr;
   device->dispatch_table.GetImageMemoryRequirements(device->dispatch_handle, b->bcn_image, &mr);
   b->memory = bcn_alloc_memory(device, &mr, false);
   if (!b->memory)
      return VK_NULL_HANDLE;
   device->dispatch_table.BindImageMemory(device->dispatch_handle, b->bcn_image, b->memory, 0);
   return b->bcn_image;
}

static const char *const bcn_slot_names[BCN_SLOT_COUNT] = {
   "decode", "astc4x4", "astc6x6", "astc8x8", "astc-hdr", "bc1-5-encode", "bc6h-encode",
   "bc7-encode",
};

struct bcn_selftest_buf {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceAddress address;
   void *map;
   bool coherent;
};

static bool
bcn_selftest_buffer(struct wrapper_device *device, VkDeviceSize size,
                    struct bcn_selftest_buf *b)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(device->dispatch_handle, &bci, NULL,
                                           &b->buffer) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(device->dispatch_handle, b->buffer, &mr);
   VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags,
      .allocationSize = mr.size,
      .memoryTypeIndex = UINT32_MAX,
   };
   const VkPhysicalDeviceMemoryProperties *props = &device->physical->memory_properties;
   static const VkMemoryPropertyFlags pref[3] = {
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
         VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
   };
   const VkMemoryPropertyFlags skip = VK_MEMORY_PROPERTY_PROTECTED_BIT |
      VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
   for (int p = 0; p < 3 && ai.memoryTypeIndex == UINT32_MAX; p++)
      for (uint32_t i = 0; i < props->memoryTypeCount && ai.memoryTypeIndex == UINT32_MAX; i++) {
         VkMemoryPropertyFlags f = props->memoryTypes[i].propertyFlags;
         if ((mr.memoryTypeBits & (1u << i)) && (f & pref[p]) == pref[p] && !(f & skip))
            ai.memoryTypeIndex = i;
      }
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(device->dispatch_handle, &ai, NULL,
                                             &b->memory) != VK_SUCCESS)
      return false;
   b->coherent = props->memoryTypes[ai.memoryTypeIndex].propertyFlags &
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   if (device->dispatch_table.BindBufferMemory(device->dispatch_handle, b->buffer, b->memory,
                                               0) != VK_SUCCESS)
      return false;
   if (device->dispatch_table.MapMemory(device->dispatch_handle, b->memory, 0, VK_WHOLE_SIZE,
                                        0, &b->map) != VK_SUCCESS)
      return false;
   memset(b->map, 0, size);
   VkBufferDeviceAddressInfo dai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = b->buffer,
   };
   b->address = device->dispatch_table.GetBufferDeviceAddress(device->dispatch_handle, &dai);
   return true;
}

static void
bcn_selftest_sync(struct wrapper_device *device, const struct bcn_selftest_buf *b,
                  const int *todo, int n, bool flush, bool all)
{
   VkMappedMemoryRange r[BCN_SLOT_COUNT];
   uint32_t c = 0;
   for (int k = 0; k < n; k++)
      if (b[todo[k]].map && (all || !b[todo[k]].coherent))
         r[c++] = (VkMappedMemoryRange) {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = b[todo[k]].memory,
            .size = VK_WHOLE_SIZE,
         };
   if (!c)
      return;
   if (flush)
      device->dispatch_table.FlushMappedMemoryRanges(device->dispatch_handle, c, r);
   else
      device->dispatch_table.InvalidateMappedMemoryRanges(device->dispatch_handle, c, r);
}

static uint32_t
bcn_selftest_zero(const struct bcn_selftest_buf *out, const int *todo, int n)
{
   uint32_t mask = 0;
   for (int k = 0; k < n; k++) {
      static const uint8_t zero[16];
      if (!memcmp(out[todo[k]].map, zero, sizeof(zero)))
         mask |= 1u << todo[k];
   }
   return mask;
}

static size_t
bcn_selftest_names(char *s, size_t size, uint32_t mask)
{
   size_t at = 0;
   for (int i = 0; i < BCN_SLOT_COUNT; i++)
      if ((mask & (1u << i)) && at < size)
         at += snprintf(s + at, size - at, " %s", bcn_slot_names[i]);
   if (!mask && size)
      at = snprintf(s, size, " none");
   return MIN2(at, size ? size - 1 : 0);
}

static void
bcn_selftest_push(struct wrapper_device *device, VkCommandBuffer cb, VkPipeline pipe,
                  const struct bcn_pc *pc)
{
   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
   };
   device->dispatch_table.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
      0, NULL, 0, NULL);
   device->dispatch_table.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   device->dispatch_table.CmdPushConstants(cb, device->bcn_tool_layout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*pc), pc);
   device->dispatch_table.CmdDispatch(cb, 1, 1, 1);
}

/* ---- shaders, built per need ----
 *
 * Nothing is built when the device is created. The first upload, readback or
 * copy that needs a shader asks for it; a worker thread builds the shaders
 * asked for, one pipeline each (the ASTC encoder per block size, the BC
 * encoder per format class), through a VkPipelineCache kept on disk and
 * keyed by the driver's pipeline cache UUID and the wrapper build. Uploads
 * take the CPU path until theirs exist; readbacks and copies wait for theirs.
 * Each new pipeline is checked on a known block inside the next vkQueueSubmit
 * on the device's first queue family, before the application's batches; one
 * that fails is dropped (and destroyed with the device, since a command
 * buffer recorded before the check may use it). */

#define BCN_CACHE_DIR "/data/data/app.gamenative/files/imagefs/usr/cache"

static VkResult
bcn_create_pipeline_specs(struct wrapper_device *device, const uint32_t *code, size_t size,
                          VkShaderModule *module, VkPipeline *pipeline, VkPipelineCache cache,
                          const uint32_t *spec, uint32_t spec_count)
{
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkResult r = VK_SUCCESS;
   static pthread_mutex_t module_lock = PTHREAD_MUTEX_INITIALIZER;
   pthread_mutex_lock(&module_lock);
   if (!*module)
      r = device->dispatch_table.CreateShaderModule(device->dispatch_handle, &smci, NULL,
                                                    module);
   pthread_mutex_unlock(&module_lock);
   if (r != VK_SUCCESS)
      return r;
   VkSpecializationMapEntry entries[2] = {
      { 0, 0, sizeof(uint32_t) },
      { 1, sizeof(uint32_t), sizeof(uint32_t) },
   };
   VkSpecializationInfo info = { spec_count, entries, spec_count * sizeof(uint32_t), spec };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = *module,
         .pName = "main",
         .pSpecializationInfo = &info,
      },
      .layout = device->bcn_tool_layout,
   };
   r = device->dispatch_table.CreateComputePipelines(device->dispatch_handle, cache, 1, &cpci,
                                                     NULL, pipeline);
   if (r != VK_SUCCESS)
      *pipeline = VK_NULL_HANDLE;
   return r;
}

static VkResult
bcn_create_pipeline_spec(struct wrapper_device *device, const uint32_t *code, size_t size,
                         VkShaderModule *module, VkPipeline *pipeline, uint32_t spec)
{
   return bcn_create_pipeline_specs(device, code, size, module, pipeline,
                                    device->bcn_pipeline_cache, &spec, 1);
}

/* The ASTC encoder for 4x4 and 6x6: 0 one invocation per block
 * (wrapper_astc_encode), 1 a workgroup per block or 2x2 blocks with shared
 * memory sums (wrapper_astc_fast), 2 the same with subgroup sums for 4x4
 * where clustered subgroup arithmetic covers 16 invocations.
 * WRAPPER_BCN_ASTC_ENCODER=group|subgroup picks one of the workgroup
 * variants; the default is the block encoder, which measured faster on a
 * Pixel 9 (Mali-G715): 4x4 loads 190 ms against 260 with subgroup sums, 6x6
 * 120 ms against 1.6-2.2 s with shared memory sums. */
static int
bcn_astc_encoder(struct wrapper_device *device, int slot)
{
   const char *e = getenv("WRAPPER_BCN_ASTC_ENCODER");
   VkPhysicalDeviceSubgroupProperties sg = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &sg,
   };
   struct wrapper_physical_device *pd = device->physical;
   pd->dispatch_table.GetPhysicalDeviceProperties2(pd->dispatch_handle, &props);
   bool sg_ok = slot == BCN_SLOT_ASTC4 && sg.subgroupSize >= 16 &&
                (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                (sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) &&
                (sg.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT);
   int enc = 0;
   if (device->bcn_astc_block_only & (1u << slot))
      e = "block";
   if (e && !strcmp(e, "group"))
      enc = 1;
   else if (e && !strcmp(e, "subgroup"))
      enc = sg_ok ? 2 : 1;
   wrapper_diag_append("[BCN] ASTC %s encoder: %s (subgroup size %u, operations 0x%x)\n",
                       slot == BCN_SLOT_ASTC4 ? "4x4" : "6x6",
                       enc == 2 ? "subgroup" : enc == 1 ? "group" : "block", sg.subgroupSize,
                       sg.supportedOperations);
   return enc;
}

static VkResult
bcn_build_slot(struct wrapper_device *device, int slot, VkPipeline *out)
{
   switch (slot) {
   case BCN_SLOT_DECODE:
      return bcn_create_pipeline_spec(device, wrapper_bcn_decode_spv,
         sizeof(wrapper_bcn_decode_spv), &device->bcn_decode_module, out, 255);
   case BCN_SLOT_ASTC4:
   case BCN_SLOT_ASTC6: {
      int enc = bcn_astc_encoder(device, slot);
      device->bcn_slot_group[slot] = enc ? (slot == BCN_SLOT_ASTC4 ? 2 : 1) : 8;
      if (enc == 2)
         return bcn_create_pipeline_spec(device, wrapper_astc_fast_sg_spv,
            sizeof(wrapper_astc_fast_sg_spv), &device->bcn_astc_fast_sg_module, out, 4);
      if (enc == 1)
         return bcn_create_pipeline_spec(device, wrapper_astc_fast_spv,
            sizeof(wrapper_astc_fast_spv), &device->bcn_astc_fast_module, out,
            4 + 2 * (slot - BCN_SLOT_ASTC4));
   }
   __attribute__((fallthrough));
   case BCN_SLOT_ASTC8:
      device->bcn_slot_group[slot] = 8;
      return bcn_create_pipeline_spec(device, wrapper_astc_encode_spv,
         sizeof(wrapper_astc_encode_spv), &device->bcn_astc_module, out,
         4 + 2 * (slot - BCN_SLOT_ASTC4));
   case BCN_SLOT_HDR:
      return bcn_create_pipeline_spec(device, wrapper_astc_hdr_encode_spv,
         sizeof(wrapper_astc_hdr_encode_spv), &device->bcn_astc_hdr_module, out, 0);
   default:
      return bcn_create_pipeline_spec(device, wrapper_bcn_encode_spv,
         sizeof(wrapper_bcn_encode_spv), &device->bcn_encode_module, out,
         slot - BCN_SLOT_ENC_BC15);
   }
}

/* Drivers that reject the full decoder or the ASTC 4x4 encoder (the PowerVR
 * 24.3 compiler returns VK_ERROR_UNKNOWN for both) get smaller variants: one
 * decoder per format, and the 4x4 encoder without the direct BC2/BC3/BC5 step.
 * They come from a fresh module and no pipeline cache. */
static VkResult
bcn_build_split(struct wrapper_device *device, int slot, VkPipeline *out)
{
   if (slot == BCN_SLOT_ASTC4 && device->bcn_slot_group[slot] == 8) {
      const uint32_t spec[2] = { 4, 0 };
      return bcn_create_pipeline_specs(device, wrapper_astc_encode_spv,
         sizeof(wrapper_astc_encode_spv), &device->bcn_split_modules[1], out, VK_NULL_HANDLE,
         spec, 2);
   }
   if (slot != BCN_SLOT_DECODE)
      return VK_ERROR_UNKNOWN;
   uint32_t built = 0;
   for (uint32_t f = 0; f < ARRAY_SIZE(device->bcn_decode_fmt); f++) {
      VkResult r = bcn_create_pipeline_specs(device, wrapper_bcn_decode_spv,
         sizeof(wrapper_bcn_decode_spv), &device->bcn_split_modules[0],
         &device->bcn_decode_fmt[f], VK_NULL_HANDLE, &f, 1);
      built |= (r == VK_SUCCESS) << f;
   }
   wrapper_diag_append("[BCN] GPU shader decode per format: built mask 0x%03x\n", built);
   device->bcn_decode_split = true;
   *out = device->bcn_decode_fmt[0];
   device->bcn_decode_fmt[0] = VK_NULL_HANDLE;
   return *out ? VK_SUCCESS : VK_ERROR_UNKNOWN;
}

static void
bcn_cache_path(struct wrapper_device *device, char *path, size_t size)
{
   const uint8_t *u = device->physical->properties2.properties.pipelineCacheUUID;
   char uuid[33];
   for (int i = 0; i < 16; i++)
      snprintf(uuid + 2 * i, 3, "%02x", u[i]);
   snprintf(path, size, BCN_CACHE_DIR "/bcn_pipelines_%s_%08x.bin", uuid,
            (uint32_t)(sizeof(wrapper_bcn_decode_spv) * 31u + sizeof(wrapper_bcn_encode_spv) * 7u +
                       sizeof(wrapper_astc_encode_spv) * 3u + sizeof(wrapper_astc_hdr_encode_spv) +
                       wrapper_bcn_encode_spv[sizeof(wrapper_bcn_encode_spv) / 8] +
                       wrapper_astc_hdr_encode_spv[sizeof(wrapper_astc_hdr_encode_spv) / 8] +
                       sizeof(wrapper_astc_fast_spv) * 5u + sizeof(wrapper_astc_fast_sg_spv) * 11u));
}

static void
bcn_cache_save_locked(struct wrapper_device *device)
{
   VkDevice d = device->dispatch_handle;
   size_t out = 0;
   void *data;
   if (!device->bcn_pipeline_cache || device->bcn_cache_tainted ||
       device->dispatch_table.GetPipelineCacheData(d, device->bcn_pipeline_cache, &out, NULL) !=
          VK_SUCCESS || !out || !(data = malloc(out)))
      return;
   char path[256], tmp[272];
   bcn_cache_path(device, path, sizeof(path));
   if (device->dispatch_table.GetPipelineCacheData(d, device->bcn_pipeline_cache, &out, data) ==
       VK_SUCCESS) {
      mkdir(BCN_CACHE_DIR, 0755);
      snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
      FILE *f = fopen(tmp, "wb");
      bool ok = f && fwrite(data, 1, out, f) == out;
      if (f)
         ok = !fclose(f) && ok;
      if (!ok || rename(tmp, path))
         remove(tmp);
   }
   free(data);
}

static void
bcn_cache_save(struct wrapper_device *device)
{
   static pthread_mutex_t save_lock = PTHREAD_MUTEX_INITIALIZER;
   pthread_mutex_lock(&save_lock);
   bcn_cache_save_locked(device);
   pthread_mutex_unlock(&save_lock);
}

#define BCN_WAIT_SECONDS 20

static void
bcn_selftest(struct wrapper_device *device, VkQueue queue, uint32_t family);

static void
bcn_gate_resolve(struct wrapper_device *device);

static void *
bcn_tools_worker(void *arg)
{
   struct wrapper_device *device = arg;
   pthread_mutex_lock(&device->bcn_tool_lock);
   while (!device->bcn_tool_stop) {
      int slot = -1;
      for (int i = 0; i < BCN_SLOT_COUNT && slot < 0; i++)
         if (device->bcn_slot_state[i] == BCN_QUEUED)
            slot = i;
      if (slot < 0) {
         pthread_cond_wait(&device->bcn_tool_cond, &device->bcn_tool_lock);
         continue;
      }
      device->bcn_slot_state[slot] = BCN_BUILDING;
      pthread_mutex_unlock(&device->bcn_tool_lock);
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      VkPipeline p = VK_NULL_HANDLE;
      VkResult r = bcn_build_slot(device, slot, &p);
      if (r != VK_SUCCESS) {
         device->bcn_cache_tainted = true;
         VkResult first = r;
         r = bcn_build_split(device, slot, &p);
         if (slot == BCN_SLOT_DECODE || slot == BCN_SLOT_ASTC4)
            wrapper_diag_append("[BCN] GPU shader %s failed (VkResult %d), split variant VkResult %d\n",
                                bcn_slot_names[slot], first, r);
      }
      clock_gettime(CLOCK_MONOTONIC, &t1);
      if (r == VK_SUCCESS && !device->bcn_cache_tainted)
         bcn_cache_save(device);
      wrapper_diag_append("[BCN] GPU shader %s built in %.0f ms (VkResult %d)\n",
                          bcn_slot_names[slot],
                          (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, r);
      pthread_mutex_lock(&device->bcn_tool_lock);
      device->bcn_pipe[slot] = p;
      device->bcn_slot_state[slot] = r == VK_SUCCESS ? BCN_BUILT : BCN_FAILED;
      __atomic_store_n(&device->bcn_tool_untested, 1, __ATOMIC_RELEASE);
      pthread_cond_broadcast(&device->bcn_tool_cond);
      if (r == VK_SUCCESS && device->bcn_queue) {
         /* checked at once on the wrapper's own queue */
         pthread_mutex_unlock(&device->bcn_tool_lock);
         simple_mtx_t *qm = device->bcn_queue_shares_query_reset ? &device->query_reset_mutex :
                            &device->bcn_queue_mutex;
         /* the same order as vkQueueSubmit: the queue's lock, then the tools' */
         simple_mtx_lock(qm);
         simple_mtx_lock(&device->bcn_gpu_mutex);
         bcn_selftest(device, device->bcn_queue, device->bcn_queue_family);
         simple_mtx_unlock(&device->bcn_gpu_mutex);
         simple_mtx_unlock(qm);
         pthread_mutex_lock(&device->bcn_tool_lock);
         pthread_cond_broadcast(&device->bcn_tool_cond);
      }
   }
   device->bcn_tool_exited++;
   pthread_cond_broadcast(&device->bcn_tool_cond);
   pthread_mutex_unlock(&device->bcn_tool_lock);
   return NULL;
}

/* Called when the device is created: records where the self-test runs. */
void
wrapper_bcn_gpu_init(struct wrapper_device *device, uint32_t family)
{
   device->bcn_tool_family = family;
   simple_mtx_init(&device->bcn_queue_mutex, mtx_plain);
   if (device->bcn_queue_reserved) {
      device->dispatch_table.GetDeviceQueue(device->dispatch_handle, device->bcn_queue_family,
                                            device->bcn_queue_index, &device->bcn_queue);
   } else if (device->query_reset_queue && !device->query_reset_shared_queue) {
      device->bcn_queue = device->query_reset_queue;
      device->bcn_queue_family = device->query_reset_queue_family;
      device->bcn_queue_shares_query_reset = true;
   }
   pthread_mutex_init(&device->bcn_tool_lock, NULL);
   pthread_cond_init(&device->bcn_tool_cond, NULL);
}

static bool
bcn_tables_create(struct wrapper_device *device)
{
   VkDevice d = device->dispatch_handle;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(wrapper_astc_tables),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(d, &bci, NULL, &device->bcn_tables) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(d, device->bcn_tables, &mr);
   VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags,
      .allocationSize = mr.size,
      .memoryTypeIndex = bcn_memory_type(device, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   void *map;
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(d, &ai, NULL, &device->bcn_tables_memory) !=
          VK_SUCCESS ||
       device->dispatch_table.BindBufferMemory(d, device->bcn_tables, device->bcn_tables_memory,
                                               0) != VK_SUCCESS ||
       device->dispatch_table.MapMemory(d, device->bcn_tables_memory, 0, VK_WHOLE_SIZE, 0,
                                        &map) != VK_SUCCESS)
      return false;
   memcpy(map, wrapper_astc_tables, sizeof(wrapper_astc_tables));
   device->dispatch_table.UnmapMemory(d, device->bcn_tables_memory);
   VkBufferDeviceAddressInfo dai = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = device->bcn_tables,
   };
   device->bcn_tables_address = device->dispatch_table.GetBufferDeviceAddress(d, &dai);
   return true;
}

/* The layout, the tables, the pipeline cache and the worker, once. */
static bool
bcn_tools_start(struct wrapper_device *device)
{
   int state = __atomic_load_n(&device->bcn_tool_state, __ATOMIC_ACQUIRE);
   if (state)
      return state > 0;
   simple_mtx_lock(&device->bcn_gpu_mutex);
   if (!device->bcn_tool_state) {
      VkDevice d = device->dispatch_handle;
      VkPushConstantRange pcr = {
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .size = sizeof(struct bcn_pc),
      };
      VkPipelineLayoutCreateInfo plci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .pushConstantRangeCount = 1,
         .pPushConstantRanges = &pcr,
      };
      char path[256];
      bcn_cache_path(device, path, sizeof(path));
      void *data = NULL;
      long len = 0;
      FILE *f = fopen(path, "rb");
      if (f) {
         if (!fseek(f, 0, SEEK_END) && (len = ftell(f)) > 0 && !fseek(f, 0, SEEK_SET) &&
             (data = malloc(len)) && fread(data, 1, len, f) != (size_t)len) {
            free(data);
            data = NULL;
         }
         fclose(f);
      }
      device->bcn_cache_warm = data != NULL;
      VkPipelineCacheCreateInfo pcci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
         .initialDataSize = data ? (size_t)len : 0,
         .pInitialData = data,
      };
      if (device->dispatch_table.CreatePipelineCache(d, &pcci, NULL,
                                                     &device->bcn_pipeline_cache) != VK_SUCCESS)
         device->bcn_pipeline_cache = VK_NULL_HANDLE;
      free(data);
      bool ok = device->bcn_bda &&
                device->dispatch_table.CreatePipelineLayout(d, &plci, NULL,
                                                            &device->bcn_tool_layout) ==
                   VK_SUCCESS &&
                bcn_tables_create(device) &&
                pthread_create(&device->bcn_tool_thread[0], NULL, bcn_tools_worker, device) == 0;
      device->bcn_tool_threads = ok;
      if (ok && sysconf(_SC_NPROCESSORS_ONLN) > 2 &&
          pthread_create(&device->bcn_tool_thread[1], NULL, bcn_tools_worker, device) == 0)
         device->bcn_tool_threads = 2;
      device->bcn_tool_thread_live = ok;
      wrapper_diag_append("[BCN] GPU shaders: %s (bda=%d, pipeline cache %ld bytes from %s; "
                          "self-test on %s)\n",
                          ok ? "built in the background" : "off, upload=cpu", device->bcn_bda,
                          device->bcn_cache_warm ? len : 0L, path,
                          device->bcn_queue_shares_query_reset ? "the query reset queue" :
                          device->bcn_queue ? "a private queue" : "the next submit");
      __atomic_store_n(&device->bcn_tool_state, ok ? 1 : -1, __ATOMIC_RELEASE);
   }
   simple_mtx_unlock(&device->bcn_gpu_mutex);
   return device->bcn_tool_state > 0;
}

/* The pipeline of a slot, queued for the worker if it was not. With the
 * wrapper's own queue every caller waits for the build and the self-test (up
 * to 20 s, once per slot): a build in flight takes a second at most, where
 * the CPU encoder takes that long for each texture. Uploads, which have the
 * CPU path, take it only when the slot failed or that wait ran out. Without
 * the wrapper's own queue the check runs at the next submit: uploads take
 * the CPU path until then, readbacks and copies take the pipeline built and
 * their dispatch is indirect with zero groups until the check passes. */
static VkPipeline
bcn_pipe(struct wrapper_device *device, int slot, bool wait)
{
   int st = __atomic_load_n(&device->bcn_slot_state[slot], __ATOMIC_ACQUIRE);
   bool built_ok = wait && !device->bcn_queue;
   if (st == BCN_TESTED || (built_ok && st == BCN_BUILT))
      return device->bcn_pipe[slot];
   if (st == BCN_FAILED || !bcn_tools_start(device))
      return VK_NULL_HANDLE;
   pthread_mutex_lock(&device->bcn_tool_lock);
   if (device->bcn_slot_state[slot] == BCN_NONE) {
      device->bcn_slot_state[slot] = BCN_QUEUED;
      pthread_cond_broadcast(&device->bcn_tool_cond);
   }
   if ((wait || device->bcn_queue) && !device->bcn_slot_late[slot]) {
      struct timespec until;
      clock_gettime(CLOCK_REALTIME, &until);
      until.tv_sec += BCN_WAIT_SECONDS;
      while (device->bcn_slot_state[slot] == BCN_QUEUED ||
             device->bcn_slot_state[slot] == BCN_BUILDING ||
             (!built_ok && device->bcn_slot_state[slot] == BCN_BUILT)) {
         if (pthread_cond_timedwait(&device->bcn_tool_cond, &device->bcn_tool_lock, &until)) {
            device->bcn_slot_late[slot] = true;
            wrapper_diag_append("[BCN] GPU shader %s not ready after %d s (state %d), not waiting "
                                "for it again\n", bcn_slot_names[slot], BCN_WAIT_SECONDS,
                                device->bcn_slot_state[slot]);
            break;
         }
      }
   }
   st = device->bcn_slot_state[slot];
   pthread_mutex_unlock(&device->bcn_tool_lock);
   return st == BCN_TESTED || (built_ok && st == BCN_BUILT) ? device->bcn_pipe[slot] : VK_NULL_HANDLE;
}

/* At device creation: the decoder and the encoders of the formats BC images
 * are stored as start building on the worker, so that the first uploads find
 * them built or close to it. */
void
wrapper_bcn_gpu_prebuild(struct wrapper_device *device)
{
   static const VkFormat formats[] = {
      VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK,
      VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC6H_UFLOAT_BLOCK,
      VK_FORMAT_BC7_UNORM_BLOCK,
   };
   if (device->physical->emulate_bcn < 2 || !wrapper_bcn_gpu_path() || !device->bcn_bda ||
       !device->bcn_queue)
      return;
   uint32_t mask = 1u << BCN_SLOT_DECODE;
   for (uint32_t i = 0; i < ARRAY_SIZE(formats); i++) {
      VkFormat stored = get_format_for_bcn(formats[i]);
      uint32_t astc = bcn_astc_block(stored);
      if (is_astc_hdr_4x4(stored))
         mask |= 1u << BCN_SLOT_HDR;
      else if (astc)
         mask |= 1u << (astc == 4 ? BCN_SLOT_ASTC4 : astc == 6 ? BCN_SLOT_ASTC6 : BCN_SLOT_ASTC8);
   }
   if (!bcn_tools_start(device))
      return;
   pthread_mutex_lock(&device->bcn_tool_lock);
   for (int i = 0; i < BCN_SLOT_COUNT; i++)
      if (((mask >> i) & 1u) && device->bcn_slot_state[i] == BCN_NONE)
         device->bcn_slot_state[i] = BCN_QUEUED;
   pthread_cond_broadcast(&device->bcn_tool_cond);
   pthread_mutex_unlock(&device->bcn_tool_lock);
}

/* Before the first image stored as ASTC 4x4: when the GPU 4x4 encoder fails
 * its build or self-test, those formats are stored decoded instead, so that
 * copies into them from uncompressed images still have a GPU path. */
void
wrapper_bcn_gpu_storage_check(struct wrapper_device *device, VkFormat format)
{
   static int decided;
   static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
   if (__atomic_load_n(&decided, __ATOMIC_ACQUIRE) ||
       !is_astc_4x4(get_format_for_bcn(format)))
      return;
   pthread_mutex_lock(&lock);
   if (!decided && device->physical->emulate_bcn > 1 && wrapper_bcn_gpu_path() &&
       device->bcn_bda && device->bcn_queue) {
      bcn_pipe(device, BCN_SLOT_ASTC4, true);
      if (__atomic_load_n(&device->bcn_slot_state[BCN_SLOT_ASTC4], __ATOMIC_ACQUIRE) ==
          BCN_FAILED) {
         wrapper_bcn_store_4x4_decoded();
         wrapper_diag_append("[BCN] ASTC 4x4 encoder unusable on this GPU: BC formats stored "
                             "as 4x4 are stored decoded\n");
      }
   }
   __atomic_store_n(&decided, 1, __ATOMIC_RELEASE);
   pthread_mutex_unlock(&lock);
}

static int
bcn_encode_slot(int format_id)
{
   return format_id == 8 || format_id == 9 ? BCN_SLOT_ENC_BC6H :
          format_id == 10 ? BCN_SLOT_ENC_BC7 : BCN_SLOT_ENC_BC15;
}

/* A command buffer recorded before the self-test may still use it: it is
 * destroyed with the device. */
static void
bcn_retire(struct wrapper_device *device, int slot)
{
   VkPipeline p = device->bcn_pipe[slot];
   for (uint32_t i = 0; p && i < ARRAY_SIZE(device->bcn_retired); i++) {
      if (!device->bcn_retired[i]) {
         device->bcn_retired[i] = p;
         break;
      }
   }
   device->bcn_pipe[slot] = VK_NULL_HANDLE;
   __atomic_store_n(&device->bcn_slot_state[slot], BCN_FAILED, __ATOMIC_RELEASE);
}

/* BC3: alpha 255, colour yellow and blue with indices 0 1 2 3 on every row */
static const uint8_t bcn_selftest_bc3[16] = {
   0xff, 0xff, 0, 0, 0, 0, 0, 0, 0xe0, 0xff, 0x1f, 0x00, 0xe4, 0xe4, 0xe4, 0xe4,
};

static bool
bcn_direct_enabled(void)
{
   const char *e = getenv("WRAPPER_BCN_DIRECT");
   return !e || strcmp(e, "0");
}

static VkResult
bcn_selftest_run(struct wrapper_device *device, VkQueue queue, VkCommandBuffer cb,
                 VkFence fence)
{
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cb,
   };
   VkResult r = device->dispatch_table.QueueSubmit(queue, 1, &si, fence);
   if (r != VK_SUCCESS)
      return r;
   r = device->dispatch_table.WaitForFences(device->dispatch_handle, 1, &fence, VK_TRUE,
                                            2000000000ull);
   if (r == VK_TIMEOUT) {
      wrapper_diag_append("[BCN] GPU self-test fence not signalled after 2 s, waiting up to %d s "
                          "more\n", BCN_WAIT_SECONDS);
      r = device->dispatch_table.WaitForFences(device->dispatch_handle, 1, &fence, VK_TRUE,
                                               BCN_WAIT_SECONDS * 1000000000ull);
   }
   return r;
}

/* Runs each pipeline built since the last check once on a known input and
 * checks the result on the CPU. WRAPPER_BCN_GPU_FAIL=decode,astc,hdr,bc drops
 * them on purpose. */
static void
bcn_selftest(struct wrapper_device *device, VkQueue queue, uint32_t family)
{
   const char *force = getenv("WRAPPER_BCN_GPU_FAIL");
   int todo[BCN_SLOT_COUNT], n = 0;
   pthread_mutex_lock(&device->bcn_tool_lock);
   __atomic_store_n(&device->bcn_tool_untested, 0, __ATOMIC_RELEASE);
   for (int i = 0; i < BCN_SLOT_COUNT; i++)
      if (device->bcn_slot_state[i] == BCN_BUILT)
         todo[n++] = i;
   pthread_mutex_unlock(&device->bcn_tool_lock);
   if (!n)
      return;

   struct timespec t0, t1;
   clock_gettime(CLOCK_MONOTONIC, &t0);
   struct bcn_selftest_buf in[BCN_SLOT_COUNT] = { 0 }, out[BCN_SLOT_COUNT] = { 0 };
   VkCommandPool pool = VK_NULL_HANDLE;
   VkCommandBuffer cb = VK_NULL_HANDLE;
   VkFence fence = VK_NULL_HANDLE;
   bool ran = false, busy = false;
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   bool bufs = true;
   for (int k = 0; k < n && bufs; k++)
      bufs = bcn_selftest_buffer(device, 256, &in[todo[k]]) &&
             bcn_selftest_buffer(device, 256, &out[todo[k]]);
   if (bufs &&
       device->dispatch_table.CreateCommandPool(device->dispatch_handle, &cpci, NULL, &pool) ==
          VK_SUCCESS &&
       (cbai.commandPool = pool,
        device->dispatch_table.AllocateCommandBuffers(device->dispatch_handle, &cbai, &cb) ==
           VK_SUCCESS) &&
       device->dispatch_table.CreateFence(device->dispatch_handle, &fci, NULL, &fence) ==
          VK_SUCCESS) {
      VkCommandBufferBeginInfo bi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      };
      device->dispatch_table.BeginCommandBuffer(cb, &bi);
      for (int k = 0; k < n; k++) {
         int s = todo[k];
         struct bcn_pc pc = {
            .src = { (uint32_t)in[s].address, (uint32_t)(in[s].address >> 32) },
            .dst = { (uint32_t)out[s].address, (uint32_t)(out[s].address >> 32) },
            .src_row_pitch = 16, .src_slice_pitch = 64, .dst_row_pitch = 16,
            .dst_slice_pitch = 16, .blocks_x = 1, .blocks_y = 1, .width = 4, .height = 4,
            .tables = { (uint32_t)device->bcn_tables_address,
                        (uint32_t)(device->bcn_tables_address >> 32) },
         };
         uint8_t *src = in[s].map;
         if (s == BCN_SLOT_DECODE) {
            /* BC1: red and blue endpoints, indices 0 1 2 3 on every row */
            static const uint8_t block[8] = { 0x00, 0xf8, 0x1f, 0x00, 0xe4, 0xe4, 0xe4, 0xe4 };
            memcpy(src, block, 8);
            pc.src_row_pitch = 8;
            pc.src_slice_pitch = 8;
            pc.dst_slice_pitch = 64;
            pc.format = 0;
         } else if (s == BCN_SLOT_HDR || s == BCN_SLOT_ENC_BC6H) {
            /* red rising from 1/16 to 12 across the block, green 1, blue 1/4 */
            uint16_t *h = (uint16_t *)src;
            for (int i = 0; i < 16; i++) {
               h[i * 4] = (uint16_t)(0x2c00 + i * 0x0200);
               h[i * 4 + 1] = 0x3c00;
               h[i * 4 + 2] = 0x3400;
               h[i * 4 + 3] = 0x3c00;
            }
            pc.src_row_pitch = 32;
            pc.src_slice_pitch = 128;
            pc.format = s == BCN_SLOT_HDR ? 4 : 8;
         } else {
            for (int i = 0; i < 16; i++) {
               src[i * 4] = (uint8_t)(i * 17);
               src[i * 4 + 1] = (uint8_t)(255 - i * 17);
               src[i * 4 + 2] = (uint8_t)((i & 3) * 85);
               src[i * 4 + 3] = 255;
            }
            pc.format = s <= BCN_SLOT_ASTC8 ? 4 + 2 * (s - BCN_SLOT_ASTC4) :
                        s == BCN_SLOT_ENC_BC7 ? 10 : 0;
         }
         bcn_selftest_push(device, cb, device->bcn_pipe[s], &pc);
         if (s == BCN_SLOT_ENC_BC7) {
            VkDeviceAddress da = out[s].address + 16;
            pc.dst[0] = (uint32_t)da;
            pc.dst[1] = (uint32_t)(da >> 32);
            pc.flags = 8;
            bcn_selftest_push(device, cb, device->bcn_pipe[s], &pc);
         }
         if (s == BCN_SLOT_ASTC4 && device->bcn_slot_group[s] == 8) {
            memcpy(src + 64, bcn_selftest_bc3, 16);
            VkDeviceAddress ba = in[s].address + 64, da = out[s].address + 16;
            pc.dst[0] = (uint32_t)da;
            pc.dst[1] = (uint32_t)(da >> 32);
            pc.flags = 4 | (3 << 8);
            pc.bc[0] = (uint32_t)ba;
            pc.bc[1] = (uint32_t)(ba >> 32);
            bcn_selftest_push(device, cb, device->bcn_pipe[s], &pc);
         }
         VkPipeline bc2_decoder = s != BCN_SLOT_DECODE ? VK_NULL_HANDLE :
                                  device->bcn_decode_split ? device->bcn_decode_fmt[2] :
                                  device->bcn_pipe[s];
         if (bc2_decoder) {
            /* BC2: explicit alpha 0..15, red and blue endpoints, decoded twice */
            static const uint8_t bc2[16] = { 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
                                             0x00, 0xf8, 0x1f, 0x00, 0x1b, 0x6c, 0xb1, 0xe4 };
            memcpy(src + 64, bc2, 16);
            VkDeviceAddress ba = in[s].address + 64;
            for (int r = 0; r < 2; r++) {
               VkDeviceAddress da = out[s].address + 64 + 64 * r;
               struct bcn_pc q = pc;
               q.src[0] = (uint32_t)ba;
               q.src[1] = (uint32_t)(ba >> 32);
               q.dst[0] = (uint32_t)da;
               q.dst[1] = (uint32_t)(da >> 32);
               q.src_row_pitch = 16;
               q.src_slice_pitch = 16;
               q.dst_row_pitch = 16;
               q.dst_slice_pitch = 64;
               q.format = 2;
               q.flags = 0;
               bcn_selftest_push(device, cb, bc2_decoder, &q);
            }
         }
         if (s == BCN_SLOT_ASTC4) {
            /* varying alpha (the BC2 case), encoded twice */
            uint8_t *t = src + 128;
            for (int i = 0; i < 16; i++) {
               t[i * 4] = (uint8_t)(i * 17);
               t[i * 4 + 1] = (uint8_t)(255 - i * 15);
               t[i * 4 + 2] = (uint8_t)((i & 3) * 80);
               t[i * 4 + 3] = (uint8_t)(40 + i * 13);
            }
            VkDeviceAddress ta = in[s].address + 128;
            for (int r = 0; r < 2; r++) {
               VkDeviceAddress da = out[s].address + 32 + 16 * r;
               struct bcn_pc q = pc;
               q.src[0] = (uint32_t)ta;
               q.src[1] = (uint32_t)(ta >> 32);
               q.dst[0] = (uint32_t)da;
               q.dst[1] = (uint32_t)(da >> 32);
               q.flags = 0;
               q.bc[0] = q.bc[1] = 0;
               bcn_selftest_push(device, cb, device->bcn_pipe[s], &q);
            }
         }
      }
      VkMemoryBarrier mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      device->dispatch_table.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
      device->dispatch_table.EndCommandBuffer(cb);
      bcn_selftest_sync(device, in, todo, n, true, false);
      VkResult r = bcn_selftest_run(device, queue, cb, fence);
      busy = r == VK_TIMEOUT;
      ran = r == VK_SUCCESS;
      if (ran) {
         bcn_selftest_sync(device, out, todo, n, false, false);
         uint32_t zero = bcn_selftest_zero(out, todo, n), stale = 0, again = 0;
         if (zero) {
            bcn_selftest_sync(device, out, todo, n, false, true);
            stale = bcn_selftest_zero(out, todo, n);
            again = stale;
            if (stale &&
                device->dispatch_table.ResetFences(device->dispatch_handle, 1, &fence) ==
                   VK_SUCCESS) {
               r = bcn_selftest_run(device, queue, cb, fence);
               busy = r == VK_TIMEOUT;
               ran = r == VK_SUCCESS;
               if (ran) {
                  bcn_selftest_sync(device, out, todo, n, false, true);
                  again = bcn_selftest_zero(out, todo, n);
               }
            }
            char a[96], b[96], c[96];
            bcn_selftest_names(a, sizeof(a), zero);
            bcn_selftest_names(b, sizeof(b), stale);
            bcn_selftest_names(c, sizeof(c), again);
            wrapper_diag_append("[BCN] GPU self-test zero blocks from%s; after invalidate%s; "
                                "after a second run%s (VkResult %d)\n", a, b, c, r);
         }
      }
   }
   clock_gettime(CLOCK_MONOTONIC, &t1);
   char report[768];
   size_t at = 0;
   for (int k = 0; k < n; k++) {
      int s = todo[k];
      const uint8_t *o = ran ? out[s].map : NULL;
      bool ok = false;
      if (o && s == BCN_SLOT_DECODE) {
         static const uint8_t pal[4][4] = {
            { 255, 0, 0, 255 }, { 0, 0, 255, 255 }, { 170, 0, 85, 255 }, { 85, 0, 170, 255 },
         };
         ok = true;
         for (int i = 0; i < 16; i++)
            ok = ok && !memcmp(o + i * 4, pal[i & 3], 4);
         if (ok && (device->bcn_decode_split ? device->bcn_decode_fmt[2] : device->bcn_pipe[s])) {
            uint8_t want[64];
            bool bc2_ok = true;
            wrapper_bcn_decode_bc2_block(in[s].map + 64, want);
            for (int i = 0; i < 64 && bc2_ok; i++)
               bc2_ok = o[64 + i] == o[128 + i] && abs((int)o[64 + i] - (int)want[i]) <= 1;
            if (!bc2_ok) {
               __atomic_or_fetch(&device->bcn_decode_bad, 1u << 2, __ATOMIC_RELEASE);
               at += snprintf(report + at, sizeof(report) - MIN2(at, sizeof(report)),
                              " decode-bc2=off");
            }
         }
      } else if (o && (s == BCN_SLOT_HDR || s <= BCN_SLOT_ASTC8)) {
         uint32_t w0;
         memcpy(&w0, o, 4);
         uint32_t cem = (w0 >> 13) & 15;
         ok = (w0 & 0x1ff) != 0x1fc && ((w0 >> 11) & 3) == 0 &&
              (s == BCN_SLOT_HDR ? (cem == 7 || cem == 11) : cem >= 8);
      } else if (o) {
         for (int i = 0; i < 8; i++)
            ok = ok || o[i] != 0;
      }
      bool direct = false, mode6 = false;
      float bc7_mse = 0.0f, bc7_mse6 = 0.0f;
      if (ok && s == BCN_SLOT_ENC_BC7) {
         /* the gradient decodes at 470.9 with every mode and 1928.6 with mode 6
          * alone on the reference GPU */
         bc7_mse = wrapper_bcn_bc7_mse(o, in[s].map);
         bc7_mse6 = wrapper_bcn_bc7_mse(o + 16, in[s].map);
         if (bc7_mse > 600.0f) {
            mode6 = true;
            ok = bc7_mse6 < 2200.0f;
         }
      }
      float rgba_mse = 0.0f;
      bool repeat = true;
      if (ok && s == BCN_SLOT_ASTC4) {
         /* the block a correct build writes (Mali-G715 r48, PowerVR 24.3) */
         static const uint32_t golden[4] = { 0xffc10253, 0xb57f7460, 0xd9d04b38, 0x01256a2b };
         repeat = !memcmp(o + 32, o + 48, 16);
         rgba_mse = wrapper_astc4_block_mse(o + 32, in[s].map + 128);
         float golden_mse = wrapper_astc4_block_mse((const uint8_t *)golden, in[s].map + 128);
         ok = repeat && (!memcmp(o + 32, golden, 16) ||
                         (rgba_mse >= 0.0f && golden_mse >= 0.0f &&
                          rgba_mse <= golden_mse * 1.25f + 10.0f));
      }
      if (ok && s == BCN_SLOT_ASTC4 && device->bcn_slot_group[s] == 8) {
         uint8_t want[16];
         wrapper_bcn_astc4_direct(VK_FORMAT_BC3_UNORM_BLOCK, bcn_selftest_bc3, want);
         direct = !memcmp(o + 16, want, 16) && !(force && strstr(force, "direct"));
      }
      const char *tag = s == BCN_SLOT_DECODE ? "decode" : s == BCN_SLOT_HDR ? "hdr" :
                        s <= BCN_SLOT_ASTC8 ? "astc" : "bc";
      if (force && strstr(force, tag))
         ok = false;
      bool retry = false;
      pthread_mutex_lock(&device->bcn_tool_lock);
      if (ok) {
         if (s == BCN_SLOT_ASTC4)
            device->bcn_direct = direct && bcn_direct_enabled();
         if (s == BCN_SLOT_ENC_BC7)
            device->bcn_bc7_mode6 = mode6;
         device->bcn_slot_state[s] = BCN_TESTED;
      } else {
         bcn_retire(device, s);
         /* a workgroup encoder that fails falls back to the block encoder */
         if ((s == BCN_SLOT_ASTC4 || s == BCN_SLOT_ASTC6) && device->bcn_slot_group[s] != 8 &&
             !(force && strstr(force, tag))) {
            device->bcn_astc_block_only |= 1u << s;
            device->bcn_slot_state[s] = BCN_QUEUED;
            pthread_cond_broadcast(&device->bcn_tool_cond);
            retry = true;
         }
      }
      pthread_mutex_unlock(&device->bcn_tool_lock);
      uint32_t w[4] = { 0 };
      if (o)
         memcpy(w, o, 16);
      at += snprintf(report + at, sizeof(report) - MIN2(at, sizeof(report)), ok ? " %s=ok" :
                     " %s=off(%08x %08x %08x %08x%s)", bcn_slot_names[s], w[0], w[1], w[2], w[3],
                     retry ? ", block encoder next" : "");
      if (o && s == BCN_SLOT_ASTC4 && at < sizeof(report))
         at += snprintf(report + at, sizeof(report) - at, " (rgba mse %.1f%s)", rgba_mse,
                        repeat ? "" : ", differs between two runs");
      if (o && s == BCN_SLOT_ENC_BC7 && at < sizeof(report))
         at += snprintf(report + at, sizeof(report) - at, " (mse %.1f, mode 6 alone %.1f%s; %08x %08x %08x %08x)",
                        bc7_mse, bc7_mse6, mode6 ? ", two-subset modes off" : "", w[0], w[1], w[2], w[3]);
      if (ok && s == BCN_SLOT_ASTC4 && device->bcn_slot_group[s] == 8 && at < sizeof(report)) {
         uint32_t dw[4];
         memcpy(dw, o + 16, 16);
         at += snprintf(report + at, sizeof(report) - at, direct ? " direct=ok" :
                        " direct=off(%08x %08x %08x %08x)", dw[0], dw[1], dw[2], dw[3]);
      }
      if (at >= sizeof(report))
         at = sizeof(report) - 1;
   }
   bcn_gate_resolve(device);
   wrapper_diag_append("[BCN] GPU self-test ran=%d%s (%.0f ms)\n", ran, report,
                       (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6);

   if (busy)
      return;
   if (fence)
      device->dispatch_table.DestroyFence(device->dispatch_handle, fence, NULL);
   if (pool)
      device->dispatch_table.DestroyCommandPool(device->dispatch_handle, pool, NULL);
   for (int i = 0; i < BCN_SLOT_COUNT; i++) {
      struct bcn_selftest_buf *b[2] = { &in[i], &out[i] };
      for (int j = 0; j < 2; j++) {
         if (b[j]->buffer)
            device->dispatch_table.DestroyBuffer(device->dispatch_handle, b[j]->buffer, NULL);
         if (b[j]->memory)
            device->dispatch_table.FreeMemory(device->dispatch_handle, b[j]->memory, NULL);
      }
   }
}

/* Inside vkQueueSubmit, where the application keeps the queue to itself. */
void
wrapper_bcn_gpu_submit(struct wrapper_queue *queue)
{
   struct wrapper_device *device = queue->device;
   if (!__atomic_load_n(&device->bcn_tool_untested, __ATOMIC_ACQUIRE) ||
       queue->vk.queue_family_index != device->bcn_tool_family) {
      /* a test running on another queue fills gated counts before this
       * submit's work can read them */
      if (__atomic_load_n(&device->bcn_gate_pending_count, __ATOMIC_ACQUIRE)) {
         simple_mtx_lock(&device->bcn_gpu_mutex);
         simple_mtx_unlock(&device->bcn_gpu_mutex);
      }
      return;
   }
   simple_mtx_lock(&device->bcn_gpu_mutex);
   bcn_selftest(device, queue->dispatch_handle, device->bcn_tool_family);
   simple_mtx_unlock(&device->bcn_gpu_mutex);
}

void
wrapper_bcn_gpu_finish(struct wrapper_device *device)
{
   VkDevice d = device->dispatch_handle;
   if (device->bcn_tool_thread_live) {
      struct timespec until;
      clock_gettime(CLOCK_REALTIME, &until);
      until.tv_sec += 15;
      pthread_mutex_lock(&device->bcn_tool_lock);
      device->bcn_tool_stop = true;
      pthread_cond_broadcast(&device->bcn_tool_cond);
      while (device->bcn_tool_exited < device->bcn_tool_threads &&
             !pthread_cond_timedwait(&device->bcn_tool_cond, &device->bcn_tool_lock, &until))
         ;
      bool exited = device->bcn_tool_exited == device->bcn_tool_threads;
      pthread_mutex_unlock(&device->bcn_tool_lock);
      device->bcn_tool_thread_live = false;
      if (!exited) {
         wrapper_diag_append("[BCN] GPU shader build still running at device destruction, "
                             "left to the process exit\n");
         for (uint32_t i = 0; i < device->bcn_tool_threads; i++)
            pthread_detach(device->bcn_tool_thread[i]);
         return;
      }
      for (uint32_t i = 0; i < device->bcn_tool_threads; i++)
         pthread_join(device->bcn_tool_thread[i], NULL);
   }
   for (int i = 0; i < BCN_SLOT_COUNT; i++)
      if (device->bcn_pipe[i])
         device->dispatch_table.DestroyPipeline(d, device->bcn_pipe[i], NULL);
   for (uint32_t i = 0; i < ARRAY_SIZE(device->bcn_decode_fmt); i++)
      if (device->bcn_decode_fmt[i])
         device->dispatch_table.DestroyPipeline(d, device->bcn_decode_fmt[i], NULL);
   for (uint32_t i = 0; i < ARRAY_SIZE(device->bcn_split_modules); i++)
      if (device->bcn_split_modules[i])
         device->dispatch_table.DestroyShaderModule(d, device->bcn_split_modules[i], NULL);
   for (uint32_t i = 0; i < ARRAY_SIZE(device->bcn_retired); i++)
      if (device->bcn_retired[i])
         device->dispatch_table.DestroyPipeline(d, device->bcn_retired[i], NULL);
   VkShaderModule modules[6] = { device->bcn_decode_module, device->bcn_encode_module,
                                 device->bcn_astc_module, device->bcn_astc_hdr_module,
                                 device->bcn_astc_fast_module, device->bcn_astc_fast_sg_module };
   for (int i = 0; i < 6; i++)
      if (modules[i])
         device->dispatch_table.DestroyShaderModule(d, modules[i], NULL);
   if (device->bcn_gate)
      device->dispatch_table.DestroyBuffer(d, device->bcn_gate, NULL);
   if (device->bcn_gate_memory)
      device->dispatch_table.FreeMemory(d, device->bcn_gate_memory, NULL);
   if (device->bcn_tables)
      device->dispatch_table.DestroyBuffer(d, device->bcn_tables, NULL);
   if (device->bcn_tables_memory)
      device->dispatch_table.FreeMemory(d, device->bcn_tables_memory, NULL);
   if (device->bcn_pipeline_cache)
      device->dispatch_table.DestroyPipelineCache(d, device->bcn_pipeline_cache, NULL);
   if (device->bcn_tool_layout)
      device->dispatch_table.DestroyPipelineLayout(d, device->bcn_tool_layout, NULL);
   pthread_cond_destroy(&device->bcn_tool_cond);
   pthread_mutex_destroy(&device->bcn_tool_lock);
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
      if (p->layout == layout && p->stages == stages && p->offset == offset && p->size == size)
         break;
   }
   if (i == wcb->push_count && wcb->push_count < ARRAY_SIZE(wcb->push))
      wcb->push_count++;
   else if (i == wcb->push_count)
      i = 0;
   memmove(&wcb->push[i], &wcb->push[i + 1], (wcb->push_count - 1 - i) * sizeof(wcb->push[0]));
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

static void
bcn_image_barrier(struct wrapper_command_buffer *wcb, VkImage image, uint32_t layers,
                  VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst)
{
   VkImageMemoryBarrier ib = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src,
      .dstAccessMask = dst,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers },
   };
   wcb->device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
      1, &ib);
}

/* Runs one of the tool shaders in the application's command buffer and puts
 * back the compute pipeline and push constants the application had set. */
#define BCN_GATE_ENTRIES 1024

static bool
bcn_gate_create(struct wrapper_device *device)
{
   VkDevice d = device->dispatch_handle;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = BCN_GATE_ENTRIES * 3 * sizeof(uint32_t),
      .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   if (device->dispatch_table.CreateBuffer(d, &bci, NULL, &device->bcn_gate) != VK_SUCCESS)
      return false;
   VkMemoryRequirements mr;
   device->dispatch_table.GetBufferMemoryRequirements(d, device->bcn_gate, &mr);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = bcn_memory_type(device, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   void *map;
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(d, &ai, NULL, &device->bcn_gate_memory) !=
          VK_SUCCESS ||
       device->dispatch_table.BindBufferMemory(d, device->bcn_gate, device->bcn_gate_memory,
                                               0) != VK_SUCCESS ||
       device->dispatch_table.MapMemory(d, device->bcn_gate_memory, 0, VK_WHOLE_SIZE, 0,
                                        &map) != VK_SUCCESS)
      return false;
   device->bcn_gate_map = map;
   return true;
}

/* Counts of the gated dispatches whose pipeline has passed its self-test.
 * Those of a failed pipeline stay zero; those of a pipeline not tested yet
 * wait for a later test. Called with bcn_gpu_mutex held, after a self-test. */
/* cond 1: only with every BC7 mode, 2: only with BC7 mode 6 alone */
static bool
bcn_gate_wanted(struct wrapper_device *device, uint32_t cond)
{
   return !cond || (cond == 2) == device->bcn_bc7_mode6;
}

static void
bcn_gate_resolve(struct wrapper_device *device)
{
   uint32_t kept = 0;
   for (uint32_t i = 0; i < device->bcn_gate_pending_count; i++) {
      uint32_t *p = device->bcn_gate_pending[i];
      int st = __atomic_load_n(&device->bcn_slot_state[p[1]], __ATOMIC_ACQUIRE);
      if (st == BCN_TESTED) {
         if (bcn_gate_wanted(device, p[5]))
            memcpy(device->bcn_gate_map + p[0] * 3, p + 2, 3 * sizeof(uint32_t));
      } else if (st != BCN_FAILED) {
         memmove(device->bcn_gate_pending[kept++], p, 6 * sizeof(uint32_t));
      }
   }
   device->bcn_gate_pending_count = kept;
}

static void
bcn_gate_dispatch(struct wrapper_command_buffer *wcb, int slot, uint32_t x, uint32_t y,
                  uint32_t z, uint32_t cond)
{
   struct wrapper_device *device = wcb->device;
   simple_mtx_lock(&device->bcn_gpu_mutex);
   if ((!device->bcn_gate_map && !bcn_gate_create(device)) ||
       device->bcn_gate_used >= BCN_GATE_ENTRIES ||
       device->bcn_gate_pending_count >= ARRAY_SIZE(device->bcn_gate_pending)) {
      simple_mtx_unlock(&device->bcn_gpu_mutex);
      static bool logged;
      if (!logged)
         wrapper_diag_append("[BCN] dispatch of an untested %s shader dropped\n",
                             bcn_slot_names[slot]);
      logged = true;
      return;
   }
   uint32_t e = device->bcn_gate_used++;
   uint32_t counts[3] = { x, y, z };
   if (__atomic_load_n(&device->bcn_slot_state[slot], __ATOMIC_ACQUIRE) == BCN_TESTED) {
      if (bcn_gate_wanted(device, cond))
         memcpy(device->bcn_gate_map + e * 3, counts, sizeof(counts));
      else
         memset(device->bcn_gate_map + e * 3, 0, sizeof(counts));
   } else {
      memset(device->bcn_gate_map + e * 3, 0, sizeof(counts));
      uint32_t *p = device->bcn_gate_pending[device->bcn_gate_pending_count++];
      p[0] = e;
      p[1] = (uint32_t)slot;
      memcpy(p + 2, counts, sizeof(counts));
      p[5] = cond;
   }
   simple_mtx_unlock(&device->bcn_gpu_mutex);
   device->dispatch_table.CmdDispatchIndirect(wcb->dispatch_handle, device->bcn_gate,
                                              (VkDeviceSize)e * 3 * sizeof(uint32_t));
}

static void
bcn_dispatch_cond(struct wrapper_command_buffer *wcb, VkPipeline pipeline,
                  const struct bcn_pc *pc, uint32_t layers, uint32_t cond)
{
   struct wrapper_device *device = wcb->device;
   struct bcn_pc p = *pc;
   p.tables[0] = (uint32_t)device->bcn_tables_address;
   p.tables[1] = (uint32_t)(device->bcn_tables_address >> 32);
   bcn_barrier(wcb);
   device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
      VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, device->bcn_tool_layout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
   uint32_t g = 8;
   int slot = -1;
   for (int i = 0; i < BCN_SLOT_COUNT; i++)
      if (device->bcn_pipe[i] == pipeline) {
         slot = i;
         if (device->bcn_slot_group[i])
            g = device->bcn_slot_group[i];
      }
   uint32_t gx = (pc->blocks_x + g - 1) / g, gy = (pc->blocks_y + g - 1) / g;
   if (slot >= 0 && (cond ||
       __atomic_load_n(&device->bcn_slot_state[slot], __ATOMIC_ACQUIRE) != BCN_TESTED))
      bcn_gate_dispatch(wcb, slot, gx, gy, layers, cond);
   else
      device->dispatch_table.CmdDispatch(wcb->dispatch_handle, gx, gy, layers);
   bcn_barrier(wcb);
   if (wcb->compute_pipeline)
      device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
         VK_PIPELINE_BIND_POINT_COMPUTE, wcb->compute_pipeline);
   for (uint32_t i = 0; i < wcb->push_count; i++)
      device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, wcb->push[i].layout,
         wcb->push[i].stages, wcb->push[i].offset, wcb->push[i].size, wcb->push[i].data);
}

static void
bcn_dispatch(struct wrapper_command_buffer *wcb, VkPipeline pipeline, const struct bcn_pc *pc,
             uint32_t layers)
{
   bcn_dispatch_cond(wcb, pipeline, pc, layers, 0);
}

static bool
bcn_blit_supported(struct wrapper_device *device, VkFormat stored, VkFormat unpacked)
{
   VkFormatProperties s = { 0 }, u = { 0 };
   device->physical->dispatch_table.GetPhysicalDeviceFormatProperties(
      device->physical->dispatch_handle, stored, &s);
   device->physical->dispatch_table.GetPhysicalDeviceFormatProperties(
      device->physical->dispatch_handle, unpacked, &u);
   return (s.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
          (u.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
}

static uint32_t
bcn_layers(const struct wrapper_image *wi, const VkImageSubresourceLayers *sub)
{
   return sub->layerCount == VK_REMAINING_ARRAY_LAYERS ?
      wi->info.arrayLayers - sub->baseArrayLayer : sub->layerCount;
}

/* BC blocks of a region of an emulated image, encoded from its stored
 * texels, in a transient buffer: rows of nbx blocks, nby rows per layer. */
static struct wrapper_buffer *
bcn_encode_region(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                  VkImageLayout layout, const VkImageSubresourceLayers *sub,
                  VkOffset3D off, VkExtent3D ext)
{
   struct wrapper_device *device = wcb->device;
   VkFormat stored = get_format_for_bcn(wi->info.format);
   VkFormat unpacked = bcn_unpacked_format(wi->info.format);
   uint32_t layers = bcn_layers(wi, sub);
   int id = bcn_format_id(wi->info.format);
   if (id < 0 || wi->info.imageType != VK_IMAGE_TYPE_2D || !ext.width || !ext.height ||
       !bcn_pipe(device, bcn_encode_slot(id), true))
      return NULL;
   if (!bcn_blit_supported(device, stored, unpacked)) {
      static bool logged;
      if (!logged)
         WRAPPER_LOG(error, "BCn: no blit from %d to %d, cannot read back", stored, unpacked);
      logged = true;
      return NULL;
   }
   uint32_t ts = unpacked == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4;
   uint32_t nbx = (ext.width + 3) / 4, nby = (ext.height + 3) / 4;
   uint32_t bb = bcn_block_bytes(wi->info.format);

   VkImage t = bcn_transient_image(wcb, unpacked, ext.width, ext.height, layers);
   VkDeviceAddress texels_addr, blocks_addr;
   struct wrapper_buffer *texels = bcn_transient_buffer(wcb,
      (VkDeviceSize)ext.width * ext.height * ts * layers, &texels_addr);
   struct wrapper_buffer *blocks = bcn_transient_buffer(wcb,
      (VkDeviceSize)nbx * nby * bb * layers, &blocks_addr);
   if (!t || !texels || !blocks)
      return NULL;

   bcn_image_barrier(wcb, t, layers, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
   VkImageBlit blit = {
      .srcSubresource = *sub,
      .srcOffsets = { off, { off.x + (int32_t)ext.width, off.y + (int32_t)ext.height, off.z + 1 } },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers },
      .dstOffsets = { { 0, 0, 0 }, { (int32_t)ext.width, (int32_t)ext.height, 1 } },
   };
   blit.srcSubresource.layerCount = layers;
   device->dispatch_table.CmdBlitImage(wcb->dispatch_handle, wi->dispatch_handle, layout, t,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
   bcn_image_barrier(wcb, t, layers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT);
   VkBufferImageCopy unpack = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers },
      .imageExtent = { ext.width, ext.height, 1 },
   };
   device->dispatch_table.CmdCopyImageToBuffer(wcb->dispatch_handle, t,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texels->dispatch_handle, 1, &unpack);

   struct bcn_pc pc = {
      .src = { (uint32_t)texels_addr, (uint32_t)(texels_addr >> 32) },
      .dst = { (uint32_t)blocks_addr, (uint32_t)(blocks_addr >> 32) },
      .src_row_pitch = ext.width * ts,
      .src_slice_pitch = ext.width * ext.height * ts,
      .dst_row_pitch = nbx * bb,
      .dst_slice_pitch = nbx * nby * bb,
      .blocks_x = nbx,
      .blocks_y = nby,
      .format = (uint32_t)id,
      .flags = (wi->info.format == VK_FORMAT_BC5_UNORM_BLOCK && is_astc(stored) ? 1 : 0) |
               (id == 10 && device->bcn_bc7_mode6 ? 8 : 0),
      .width = ext.width,
      .height = ext.height,
   };
   VkPipeline enc = bcn_pipe(device, bcn_encode_slot(id), true);
   if (id == 10 && __atomic_load_n(&device->bcn_slot_state[BCN_SLOT_ENC_BC7],
                                   __ATOMIC_ACQUIRE) != BCN_TESTED) {
      /* which BC7 modes work is known once the self-test has run */
      pc.flags &= ~8u;
      bcn_dispatch_cond(wcb, enc, &pc, layers, 1);
      pc.flags |= 8u;
      bcn_dispatch_cond(wcb, enc, &pc, layers, 2);
   } else {
      bcn_dispatch(wcb, enc, &pc, layers);
   }
   return blocks;
}

bool
wrapper_bcn_gpu_readback(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                         VkImageLayout layout, VkBuffer dst, uint32_t count,
                         const VkBufferImageCopy *regions)
{
   struct wrapper_device *device = wcb->device;
   uint32_t bb = bcn_block_bytes(wi->info.format);
   for (uint32_t i = 0; i < count; i++) {
      const VkBufferImageCopy *c = &regions[i];
      struct wrapper_buffer *blocks = bcn_encode_region(wcb, wi, layout, &c->imageSubresource,
                                                        c->imageOffset, c->imageExtent);
      if (!blocks)
         return i > 0;
      uint32_t layers = bcn_layers(wi, &c->imageSubresource);
      uint32_t nbx = (c->imageExtent.width + 3) / 4, nby = (c->imageExtent.height + 3) / 4;
      uint32_t row_len = c->bufferRowLength ? c->bufferRowLength : c->imageExtent.width;
      uint32_t img_h = c->bufferImageHeight ? c->bufferImageHeight : c->imageExtent.height;
      VkDeviceSize row = (VkDeviceSize)((row_len + 3) / 4) * bb;
      VkDeviceSize slice = row * ((img_h + 3) / 4);
      VkDeviceSize tight = (VkDeviceSize)nbx * bb;
      for (uint32_t l = 0; l < layers; l++) {
         if (row == tight) {
            VkBufferCopy bc = { (VkDeviceSize)l * nbx * nby * bb, c->bufferOffset + l * slice,
                                tight * nby };
            device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, blocks->dispatch_handle,
                                                 dst, 1, &bc);
            continue;
         }
         for (uint32_t y = 0; y < nby; y++) {
            VkBufferCopy bc = { ((VkDeviceSize)l * nby + y) * tight,
                                c->bufferOffset + l * slice + y * row, tight };
            device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, blocks->dispatch_handle,
                                                 dst, 1, &bc);
         }
      }
   }
   return true;
}

static bool
bcn_gpu_upload_region(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                      struct wrapper_buffer *wb, VkBuffer src, VkImageLayout layout,
                      const VkBufferImageCopy *c, bool wait);

/* One vkCmdCopyImage region with an emulated BC image on one side or both.
 * Returns false when the region should go to the driver as it is. */
bool
wrapper_bcn_gpu_copy_image_region(struct wrapper_command_buffer *wcb,
                                  struct wrapper_image *src, bool src_emulated,
                                  VkImageLayout src_layout,
                                  struct wrapper_image *dst, bool dst_emulated,
                                  VkImageLayout dst_layout, const VkImageCopy *c)
{
   struct wrapper_device *device = wcb->device;

   if (src_emulated && dst_emulated) {
      VkImageSubresourceLayers sub = c->dstSubresource;
      sub.layerCount = bcn_layers(dst, &sub);
      wrapper_bcn_hq_write(device, dst, &sub, c->dstOffset, c->extent);
      /* Same stored format and whole stored blocks: the driver copies them. */
      VkFormat s = get_format_for_bcn(src->info.format);
      if (s == get_format_for_bcn(dst->info.format))
         return false;
      static bool logged;
      if (!logged)
         WRAPPER_LOG(error, "BCn: copy between BC formats stored as %d and %d is not supported",
                     s, get_format_for_bcn(dst->info.format));
      logged = true;
      return false;
   }

   if (src_emulated) {
      VkExtent3D ext = c->extent;
      struct wrapper_buffer *blocks = bcn_encode_region(wcb, src, src_layout,
                                                        &c->srcSubresource, c->srcOffset, ext);
      if (!blocks)
         return false;
      uint32_t nbx = (ext.width + 3) / 4, nby = (ext.height + 3) / 4;
      VkBufferImageCopy bic = {
         .bufferRowLength = nbx,
         .bufferImageHeight = nby,
         .imageSubresource = c->dstSubresource,
         .imageOffset = c->dstOffset,
         .imageExtent = { nbx, nby, 1 },
      };
      device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle,
         blocks->dispatch_handle, dst->dispatch_handle, dst_layout, 1, &bic);
      return true;
   }

   /* Blocks from an uncompressed image (a GPU encoder writing BC blocks through
    * a uint image): copy them out and store them as an upload of the same
    * blocks would be. */
   int id = bcn_format_id(dst->info.format);
   if (id < 0 || dst->info.imageType != VK_IMAGE_TYPE_2D ||
       !bcn_pipe(device, BCN_SLOT_DECODE, true)) {
      static bool logged;
      if (!logged)
         WRAPPER_LOG(error, "BCn: copy into BC format %d stored as %d is not supported",
                     dst->info.format, get_format_for_bcn(dst->info.format));
      logged = true;
      return false;
   }
   uint32_t layers = bcn_layers(src, &c->srcSubresource);
   uint32_t nbx = c->extent.width, nby = c->extent.height;
   uint32_t bb = bcn_block_bytes(dst->info.format);
   uint32_t m = c->dstSubresource.mipLevel;
   uint32_t lw = MAX2(1u, dst->info.extent.width >> m), lh = MAX2(1u, dst->info.extent.height >> m);
   VkDeviceAddress blocks_addr;
   struct wrapper_buffer *blocks = bcn_transient_buffer(wcb,
      (VkDeviceSize)nbx * nby * bb * layers, &blocks_addr);
   if (!blocks)
      return false;
   bcn_barrier(wcb);
   VkBufferImageCopy pack = {
      .imageSubresource = c->srcSubresource,
      .imageOffset = c->srcOffset,
      .imageExtent = { nbx, nby, 1 },
   };
   device->dispatch_table.CmdCopyImageToBuffer(wcb->dispatch_handle, src->dispatch_handle,
      src_layout, blocks->dispatch_handle, 1, &pack);
   VkBufferImageCopy up = {
      .imageSubresource = c->dstSubresource,
      .imageOffset = c->dstOffset,
      .imageExtent = { MIN2(nbx * 4, lw - (uint32_t)c->dstOffset.x),
                       MIN2(nby * 4, lh - (uint32_t)c->dstOffset.y), 1 },
   };
   if (!bcn_gpu_upload_region(wcb, dst, NULL, blocks->dispatch_handle, dst_layout, &up, true)) {
      static bool logged;
      if (!logged)
         WRAPPER_LOG(error, "BCn: GPU path cannot store a copy into format %d",
                     dst->info.format);
      logged = true;
   }
   return true;
}

static uint32_t
bcn_astc_block(VkFormat stored)
{
   return is_astc_6x6(stored) ? 6 : is_astc_8x8(stored) ? 8 : is_astc_4x4(stored) ? 4 : 0;
}

/* Whether a region of an image stored as ASTC does not cover whole ASTC
 * blocks: BC regions start on 4-texel boundaries, 6x6 and 8x8 blocks do not. */
bool
wrapper_bcn_gpu_misaligned(const struct wrapper_image *wi, const VkBufferImageCopy *c)
{
   uint32_t astc = bcn_astc_block(get_format_for_bcn(wi->info.format));
   if (!astc)
      return false;
   uint32_t m = c->imageSubresource.mipLevel;
   uint32_t lw = MAX2(1u, wi->info.extent.width >> m), lh = MAX2(1u, wi->info.extent.height >> m);
   uint32_t w = c->imageExtent.width, h = c->imageExtent.height;
   return (c->imageOffset.x % astc) || (c->imageOffset.y % astc) ||
          ((w % astc) && c->imageOffset.x + w != lw) ||
          ((h % astc) && c->imageOffset.y + h != lh);
}

/* A whole-level upload with an entry in the disk cache: the cached blocks go
 * straight into the image. */
static bool
bcn_upload_cached(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                  struct wrapper_buffer *wb, VkImageLayout layout, const VkBufferImageCopy *c,
                  VkFormat stored, uint32_t astc)
{
   struct wrapper_device *device = wcb->device;
   VkFormat format = wi->info.format;
   uint32_t m = c->imageSubresource.mipLevel;
   uint32_t w = c->imageExtent.width, h = c->imageExtent.height;
   if (!wrapper_bcn_cache_on() || !wb || wb->device_only || bcn_layers(wi, &c->imageSubresource) != 1 ||
       c->imageOffset.x || c->imageOffset.y || c->imageOffset.z ||
       w != MAX2(1u, wi->info.extent.width >> m) || h != MAX2(1u, wi->info.extent.height >> m))
      return false;
   uint32_t bb = bcn_block_bytes(format), nby = (h + 3) / 4;
   size_t row = (size_t)((w + 3) / 4) * bb;
   size_t stride = (size_t)(((c->bufferRowLength ? c->bufferRowLength : w) + 3) / 4) * bb;
   size_t len = (size_t)((w + astc - 1) / astc) * ((h + astc - 1) / astc) * 16;
   simple_mtx_lock(&device->resource_mutex);
   if (!wb->is_mapped) {
      if (device->dispatch_table.MapMemory(device->dispatch_handle, wb->memory, wb->offset,
                                           wb->size, 0, &wb->mapped_address) != VK_SUCCESS) {
         simple_mtx_unlock(&device->resource_mutex);
         return false;
      }
      wb->is_mapped = 1;
   }
   bool inside = c->bufferOffset + stride * (nby - 1) + row <= wb->size;
   simple_mtx_unlock(&device->resource_mutex);
   void *map = NULL;
   struct wrapper_buffer *t = NULL;
   uint8_t *blocks = inside ? malloc(len) : NULL;
   bool hit = blocks && wrapper_bcn_cache_load(format, stored, w, h,
      (const uint8_t *)wb->mapped_address + c->bufferOffset, row, stride, nby, blocks, len) &&
      (t = bcn_transient_host_buffer(wcb, len, &map));
   if (hit) {
      memcpy(map, blocks, len);
      VkBufferImageCopy copy = {
         .bufferRowLength = (w + astc - 1) / astc * astc,
         .bufferImageHeight = (h + astc - 1) / astc * astc,
         .imageSubresource = c->imageSubresource,
         .imageExtent = c->imageExtent,
      };
      copy.imageSubresource.layerCount = 1;
      bcn_barrier(wcb);
      device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle, t->dispatch_handle,
                                                  wi->dispatch_handle, layout, 1, &copy);
      wrapper_bcn_hq_write(device, wi, &copy.imageSubresource, copy.imageOffset,
                           copy.imageExtent);
   }
   free(blocks);
   return hit;
}

/* GPU upload of one region: the blocks are copied out of the application's
 * buffer, decoded, and either copied in as decoded texels or encoded to ASTC
 * first. A region that does not cover whole ASTC blocks is merged into the
 * texels already in the image: the covering blocks are unpacked, overwritten
 * and encoded again. */
static bool
bcn_gpu_upload_region(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                      struct wrapper_buffer *wb, VkBuffer src, VkImageLayout layout,
                      const VkBufferImageCopy *c, bool wait)
{
   struct wrapper_device *device = wcb->device;
   VkFormat format = wi->info.format;
   VkFormat stored = get_format_for_bcn(format);
   int id = bcn_format_id(format);
   bool hdr = is_astc_hdr_4x4(stored);
   bool packed = stored == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   uint32_t astc = hdr ? 4 : bcn_astc_block(stored);
   bool decoded = stored == get_decode_format_for_bcn(format);
   if (id < 0 || (!astc && !decoded && !packed) || wi->info.imageType != VK_IMAGE_TYPE_2D)
      return false;
   uint32_t w = c->imageExtent.width, h = c->imageExtent.height;
   uint32_t m = c->imageSubresource.mipLevel;
   uint32_t lw = MAX2(1u, wi->info.extent.width >> m), lh = MAX2(1u, wi->info.extent.height >> m);
   if (!w || !h)
      return false;
   if (astc && bcn_upload_cached(wcb, wi, wb, layout, c, stored, astc))
      return true;
   VkPipeline decoder = bcn_pipe(device, BCN_SLOT_DECODE, wait);
   if (decoder && device->bcn_decode_split)
      decoder = id == 0 ? decoder : device->bcn_decode_fmt[id];
   if ((__atomic_load_n(&device->bcn_decode_bad, __ATOMIC_ACQUIRE) >> id) & 1u)
      decoder = VK_NULL_HANDLE;
   VkPipeline encoder = hdr ? bcn_pipe(device, BCN_SLOT_HDR, wait) :
                        astc ? bcn_pipe(device, astc == 4 ? BCN_SLOT_ASTC4 :
                                        astc == 6 ? BCN_SLOT_ASTC6 : BCN_SLOT_ASTC8, wait) :
                        VK_NULL_HANDLE;
   if (!decoder || (astc && !encoder))
      return false;
   bool merge = astc && wrapper_bcn_gpu_misaligned(wi, c);
   /* the rectangle written: the region, or the ASTC blocks covering it */
   uint32_t x0 = c->imageOffset.x, y0 = c->imageOffset.y, ew = w, eh = h;
   if (merge) {
      x0 = c->imageOffset.x / astc * astc;
      y0 = c->imageOffset.y / astc * astc;
      ew = MIN2((c->imageOffset.x + w + astc - 1) / astc * astc, lw) - x0;
      eh = MIN2((c->imageOffset.y + h + astc - 1) / astc * astc, lh) - y0;
      if (!bcn_blit_supported(device, stored, bcn_unpacked_format(format)))
         return false;
   }

   uint32_t layers = bcn_layers(wi, &c->imageSubresource);
   uint32_t bb = bcn_block_bytes(format);
   uint32_t nbx = (w + 3) / 4, nby = (h + 3) / 4;
   uint32_t ts = hdr ? 8 : astc ? 4 : get_texel_size_for_format(stored);
   uint32_t row_len = c->bufferRowLength ? c->bufferRowLength : w;
   uint32_t img_h = c->bufferImageHeight ? c->bufferImageHeight : h;
   VkDeviceSize src_row = (VkDeviceSize)((row_len + 3) / 4) * bb;
   VkDeviceSize src_slice = src_row * ((img_h + 3) / 4);
   VkDeviceSize tight = (VkDeviceSize)nbx * bb;

   VkDeviceAddress blocks_addr, texels_addr;
   struct wrapper_buffer *blocks = bcn_transient_buffer(wcb, tight * nby * layers, &blocks_addr);
   struct wrapper_buffer *texels = bcn_transient_buffer(wcb,
      (VkDeviceSize)nbx * 4 * nby * 4 * ts * layers, &texels_addr);
   if (!blocks || !texels)
      return false;
   struct wrapper_buffer *out = NULL;
   VkDeviceAddress out_addr = 0;
   uint32_t abx = 0, aby = 0;
   if (astc) {
      abx = (w + astc - 1) / astc;
      aby = (h + astc - 1) / astc;
      out = bcn_transient_buffer(wcb, (VkDeviceSize)abx * aby * 16 * layers, &out_addr);
      if (!out)
         return false;
   }

   bcn_barrier(wcb);
   for (uint32_t l = 0; l < layers; l++) {
      if (src_row == tight) {
         VkBufferCopy bc = { c->bufferOffset + l * src_slice, (VkDeviceSize)l * tight * nby,
                             tight * nby };
         device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, src, blocks->dispatch_handle,
                                              1, &bc);
         continue;
      }
      for (uint32_t y = 0; y < nby; y++) {
         VkBufferCopy bc = { c->bufferOffset + l * src_slice + y * src_row,
                             ((VkDeviceSize)l * nby + y) * tight, tight };
         device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, src, blocks->dispatch_handle,
                                              1, &bc);
      }
   }

   struct bcn_pc pc = {
      .src = { (uint32_t)blocks_addr, (uint32_t)(blocks_addr >> 32) },
      .dst = { (uint32_t)texels_addr, (uint32_t)(texels_addr >> 32) },
      .src_row_pitch = (uint32_t)tight,
      .src_slice_pitch = (uint32_t)(tight * nby),
      .dst_row_pitch = nbx * 4 * ts,
      .dst_slice_pitch = nbx * 4 * nby * 4 * ts,
      .blocks_x = nbx,
      .blocks_y = nby,
      .format = (uint32_t)id,
      .flags = (astc && !hdr) ? 1 : packed ? 2 : 0,
   };
   bcn_dispatch(wcb, decoder, &pc, layers);

   VkBufferImageCopy copy = {
      .bufferRowLength = nbx * 4,
      .bufferImageHeight = nby * 4,
      .imageSubresource = c->imageSubresource,
      .imageOffset = c->imageOffset,
      .imageExtent = c->imageExtent,
   };
   VkBuffer from = texels->dispatch_handle;
   VkDeviceAddress enc_addr = texels_addr;
   uint32_t enc_row = nbx * 4 * ts, enc_slice = nbx * 4 * nby * 4 * ts;
   if (merge) {
      /* unpack the covering blocks, then lay the new texels over them */
      VkImage t = bcn_transient_image(wcb, bcn_unpacked_format(format), ew, eh, layers);
      VkDeviceAddress merged_addr;
      struct wrapper_buffer *merged = bcn_transient_buffer(wcb,
         (VkDeviceSize)ew * eh * 4 * layers, &merged_addr);
      if (!t || !merged)
         return false;
      VkImageSubresourceRange range = {
         VK_IMAGE_ASPECT_COLOR_BIT, m, 1, c->imageSubresource.baseArrayLayer, layers,
      };
      VkImageMemoryBarrier to_src = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = layout,
         .newLayout = layout == VK_IMAGE_LAYOUT_GENERAL ? layout : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = wi->dispatch_handle,
         .subresourceRange = range,
      };
      device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
         1, &to_src);
      bcn_image_barrier(wcb, t, layers, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
      VkImageBlit blit = {
         .srcSubresource = c->imageSubresource,
         .srcOffsets = { { (int32_t)x0, (int32_t)y0, 0 },
                         { (int32_t)(x0 + ew), (int32_t)(y0 + eh), 1 } },
         .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers },
         .dstOffsets = { { 0, 0, 0 }, { (int32_t)ew, (int32_t)eh, 1 } },
      };
      blit.srcSubresource.layerCount = layers;
      device->dispatch_table.CmdBlitImage(wcb->dispatch_handle, wi->dispatch_handle,
         to_src.newLayout, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
      VkImageMemoryBarrier back = to_src;
      back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      back.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      back.oldLayout = to_src.newLayout;
      back.newLayout = layout;
      device->dispatch_table.CmdPipelineBarrier(wcb->dispatch_handle,
         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
         1, &back);
      bcn_image_barrier(wcb, t, layers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT);
      VkBufferImageCopy unpack = {
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers },
         .imageExtent = { ew, eh, 1 },
      };
      device->dispatch_table.CmdCopyImageToBuffer(wcb->dispatch_handle, t,
         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, merged->dispatch_handle, 1, &unpack);
      bcn_barrier(wcb);
      uint32_t dx = c->imageOffset.x - x0, dy = c->imageOffset.y - y0;
      for (uint32_t l = 0; l < layers; l++)
         for (uint32_t y = 0; y < h; y++) {
            VkBufferCopy bc = {
               ((VkDeviceSize)l * nby * 4 + y) * nbx * 4 * 4,
               (((VkDeviceSize)l * eh + dy + y) * ew + dx) * 4,
               (VkDeviceSize)w * 4,
            };
            device->dispatch_table.CmdCopyBuffer(wcb->dispatch_handle, texels->dispatch_handle,
                                                 merged->dispatch_handle, 1, &bc);
         }
      enc_addr = merged_addr;
      enc_row = ew * 4;
      enc_slice = ew * eh * 4;
      w = ew;
      h = eh;
      abx = (w + astc - 1) / astc;
      aby = (h + astc - 1) / astc;
      out = bcn_transient_buffer(wcb, (VkDeviceSize)abx * aby * 16 * layers, &out_addr);
      if (!out)
         return false;
      copy.imageOffset = (VkOffset3D){ (int32_t)x0, (int32_t)y0, c->imageOffset.z };
      copy.imageExtent = (VkExtent3D){ ew, eh, 1 };
   }
   if (astc) {
      struct bcn_pc ac = {
         .src = { (uint32_t)enc_addr, (uint32_t)(enc_addr >> 32) },
         .dst = { (uint32_t)out_addr, (uint32_t)(out_addr >> 32) },
         .src_row_pitch = enc_row,
         .src_slice_pitch = enc_slice,
         .dst_row_pitch = abx * 16,
         .dst_slice_pitch = abx * aby * 16,
         .blocks_x = abx,
         .blocks_y = aby,
         .format = astc,
         .flags = format == VK_FORMAT_BC4_UNORM_BLOCK ? 1 :
                  format == VK_FORMAT_BC5_UNORM_BLOCK ? 2 : 0,
         .width = w,
         .height = h,
      };
      if (astc == 4 && !merge && (id == 2 || id == 3 || id == 6) &&
          __atomic_load_n(&device->bcn_direct, __ATOMIC_RELAXED)) {
         ac.flags |= 4 | ((uint32_t)id << 8);
         ac.bc[0] = (uint32_t)blocks_addr;
         ac.bc[1] = (uint32_t)(blocks_addr >> 32);
      }
      bcn_dispatch(wcb, encoder, &ac, layers);
      copy.bufferRowLength = abx * astc;
      copy.bufferImageHeight = aby * astc;
      from = out->dispatch_handle;
   }
   device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle, from, wi->dispatch_handle,
                                               layout, 1, &copy);
   if (astc && !merge)
      wrapper_bcn_hq_capture(wcb, wi, wb, layout, c, astc);
   else
      wrapper_bcn_hq_write(device, wi, &copy.imageSubresource, copy.imageOffset,
                           copy.imageExtent);
   return true;
}

uint32_t
wrapper_bcn_gpu_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       struct wrapper_buffer *wb, VkImageLayout layout, uint32_t count,
                       const VkBufferImageCopy *regions)
{
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   __atomic_store_n(&wrapper_bcn_upload_ms, (uint64_t)now.tv_sec * 1000u + now.tv_nsec / 1000000,
                    __ATOMIC_RELAXED);
   uint32_t done = 0;
   while (done < count && bcn_gpu_upload_region(wcb, wi, wb, wb->dispatch_handle, layout,
                                                &regions[done], wb->device_only))
      done++;
   return done;
}
