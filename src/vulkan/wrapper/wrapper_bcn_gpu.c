#include <stdlib.h>
#include <string.h>

#include "wrapper_bcn_gpu.h"
#include "wrapper_bcdec.h"
#include "wrapper_bcn_decode_spv.h"
#include "wrapper_bcn_encode_spv.h"
#include "wrapper_astc_encode_spv.h"
#include "wrapper_log.h"
#include "util/macros.h"

/* GPU work for emulated BC images. A readback, or a copy to or from another
 * image, needs BC blocks where the image holds ASTC or decoded texels: the
 * stored texels are blitted to an RGBA image (the driver decodes ASTC) and
 * BC-encoded by a compute shader; blocks copied in are decoded by another.
 * Uploads are decoded and encoded to ASTC on the GPU whenever the shaders
 * passed the self-test; the CPU path is the fallback, or forced with
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

static VkResult
bcn_create_pipeline_spec(struct wrapper_device *device, const uint32_t *code, size_t size,
                         VkShaderModule *module, VkPipeline *pipeline, uint32_t edge)
{
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkResult r = VK_SUCCESS;
   if (!*module)
      r = device->dispatch_table.CreateShaderModule(device->dispatch_handle, &smci, NULL,
                                                    module);
   if (r != VK_SUCCESS)
      return r;
   VkSpecializationMapEntry entry = { 0, 0, sizeof(uint32_t) };
   VkSpecializationInfo spec = { 1, &entry, sizeof(uint32_t), &edge };
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = *module,
         .pName = "main",
         .pSpecializationInfo = edge ? &spec : NULL,
      },
      .layout = device->bcn_tool_layout,
   };
   r = device->dispatch_table.CreateComputePipelines(device->dispatch_handle,
      VK_NULL_HANDLE, 1, &cpci, NULL, pipeline);
   if (r != VK_SUCCESS)
      *pipeline = VK_NULL_HANDLE;
   return r;
}

static VkResult
bcn_create_pipeline(struct wrapper_device *device, const uint32_t *code, size_t size,
                    VkShaderModule *module, VkPipeline *pipeline)
{
   return bcn_create_pipeline_spec(device, code, size, module, pipeline, 0);
}

struct bcn_selftest_buf {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceAddress address;
   void *map;
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
      .memoryTypeIndex = bcn_memory_type(device, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   if (ai.memoryTypeIndex == UINT32_MAX ||
       device->dispatch_table.AllocateMemory(device->dispatch_handle, &ai, NULL,
                                             &b->memory) != VK_SUCCESS)
      return false;
   device->dispatch_table.BindBufferMemory(device->dispatch_handle, b->buffer, b->memory, 0);
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

static void
bcn_disable_pipeline(struct wrapper_device *device, VkPipeline *pipeline)
{
   if (*pipeline)
      device->dispatch_table.DestroyPipeline(device->dispatch_handle, *pipeline, NULL);
   *pipeline = VK_NULL_HANDLE;
}

/* Runs each shader once on a known block on the device's first queue and
 * checks the result on the CPU. A shader that does not produce it is
 * dropped, so the work that needs it stays on the CPU path.
 * WRAPPER_BCN_GPU_FAIL=decode,bc,astc drops them on purpose. */
static void
bcn_selftest(struct wrapper_device *device, uint32_t family, char *report, size_t size)
{
   const char *force = getenv("WRAPPER_BCN_GPU_FAIL");
   struct bcn_selftest_buf blk = { 0 }, rgba = { 0 }, astc = { 0 }, bc = { 0 };
   VkCommandPool pool = VK_NULL_HANDLE;
   VkCommandBuffer cb = VK_NULL_HANDLE;
   VkFence fence = VK_NULL_HANDLE;
   bool ran = false;
   int dec_ok = 0, astc_ok = 0, bc_ok = 0;

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
   if (bcn_selftest_buffer(device, 16, &blk) && bcn_selftest_buffer(device, 64, &rgba) &&
       bcn_selftest_buffer(device, 16, &astc) && bcn_selftest_buffer(device, 16, &bc) &&
       device->dispatch_table.CreateCommandPool(device->dispatch_handle, &cpci, NULL,
                                                &pool) == VK_SUCCESS &&
       (cbai.commandPool = pool,
        device->dispatch_table.AllocateCommandBuffers(device->dispatch_handle, &cbai,
                                                      &cb) == VK_SUCCESS) &&
       device->dispatch_table.CreateFence(device->dispatch_handle, &fci, NULL,
                                          &fence) == VK_SUCCESS) {
      /* BC1: red and blue endpoints, indices 0 1 2 3 on every row */
      static const uint8_t block[8] = { 0x00, 0xf8, 0x1f, 0x00, 0xe4, 0xe4, 0xe4, 0xe4 };
      memcpy(blk.map, block, 8);
      VkCommandBufferBeginInfo bi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      device->dispatch_table.BeginCommandBuffer(cb, &bi);
      struct bcn_pc pc = {
         .src = { (uint32_t)blk.address, (uint32_t)(blk.address >> 32) },
         .dst = { (uint32_t)rgba.address, (uint32_t)(rgba.address >> 32) },
         .src_row_pitch = 8, .src_slice_pitch = 8, .dst_row_pitch = 16, .dst_slice_pitch = 64,
         .blocks_x = 1, .blocks_y = 1, .format = 0, .width = 4, .height = 4,
      };
      if (device->bcn_decode_pipeline)
         bcn_selftest_push(device, cb, device->bcn_decode_pipeline, &pc);
      struct bcn_pc ac = {
         .src = { (uint32_t)rgba.address, (uint32_t)(rgba.address >> 32) },
         .dst = { (uint32_t)astc.address, (uint32_t)(astc.address >> 32) },
         .src_row_pitch = 16, .src_slice_pitch = 64, .dst_row_pitch = 16, .dst_slice_pitch = 16,
         .blocks_x = 1, .blocks_y = 1, .format = 4, .flags = 0, .width = 4, .height = 4,
      };
      if (device->bcn_decode_pipeline && device->bcn_astc_pipeline[0])
         bcn_selftest_push(device, cb, device->bcn_astc_pipeline[0], &ac);
      struct bcn_pc ec = ac;
      ec.dst[0] = (uint32_t)bc.address;
      ec.dst[1] = (uint32_t)(bc.address >> 32);
      ec.dst_row_pitch = 8;
      ec.dst_slice_pitch = 8;
      ec.format = 0;
      if (device->bcn_decode_pipeline && device->bcn_encode_pipeline)
         bcn_selftest_push(device, cb, device->bcn_encode_pipeline, &ec);
      VkMemoryBarrier mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      device->dispatch_table.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
      device->dispatch_table.EndCommandBuffer(cb);
      VkQueue queue = VK_NULL_HANDLE;
      device->dispatch_table.GetDeviceQueue(device->dispatch_handle, family, 0, &queue);
      VkSubmitInfo si = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &cb,
      };
      ran = queue && device->dispatch_table.QueueSubmit(queue, 1, &si, fence) == VK_SUCCESS &&
            device->dispatch_table.WaitForFences(device->dispatch_handle, 1, &fence, VK_TRUE,
                                                 2000000000ull) == VK_SUCCESS;
   }
   if (ran) {
      static const uint8_t pal[4][4] = {
         { 255, 0, 0, 255 }, { 0, 0, 255, 255 }, { 170, 0, 85, 255 }, { 85, 0, 170, 255 },
      };
      const uint8_t *got = rgba.map;
      dec_ok = 1;
      for (int i = 0; i < 16; i++)
         dec_ok &= !memcmp(got + i * 4, pal[i & 3], 4);
      const uint8_t *a = astc.map;
      for (int i = 0; i < 16; i++)
         astc_ok |= a[i] != 0;
      const uint8_t *b = bc.map;
      bc_ok = (b[0] | b[1] | b[2] | b[3]) != 0;
   }
   if (!dec_ok || (force && strstr(force, "decode")))
      bcn_disable_pipeline(device, &device->bcn_decode_pipeline);
   if (!astc_ok || (force && strstr(force, "astc")))
      for (int i = 0; i < 3; i++)
         bcn_disable_pipeline(device, &device->bcn_astc_pipeline[i]);
   if (!bc_ok || (force && strstr(force, "bc")))
      bcn_disable_pipeline(device, &device->bcn_encode_pipeline);
   const uint8_t *g = ran ? rgba.map : NULL;
   snprintf(report, size, "self-test ran=%d decode=%s astc=%s bc=%s first=%02x%02x%02x%02x",
            ran, device->bcn_decode_pipeline ? "ok" : "off",
            device->bcn_astc_pipeline[0] ? "ok" : "off", device->bcn_encode_pipeline ? "ok" : "off",
            g ? g[0] : 0, g ? g[1] : 0, g ? g[2] : 0, g ? g[3] : 0);

   if (fence)
      device->dispatch_table.DestroyFence(device->dispatch_handle, fence, NULL);
   if (pool)
      device->dispatch_table.DestroyCommandPool(device->dispatch_handle, pool, NULL);
   struct bcn_selftest_buf *bufs[4] = { &blk, &rgba, &astc, &bc };
   for (int i = 0; i < 4; i++) {
      if (bufs[i]->buffer)
         device->dispatch_table.DestroyBuffer(device->dispatch_handle, bufs[i]->buffer, NULL);
      if (bufs[i]->memory)
         device->dispatch_table.FreeMemory(device->dispatch_handle, bufs[i]->memory, NULL);
   }
}

/* Creates the shaders once per device and checks them on a known block;
 * each one that fails only disables the work that needs it. */
void
wrapper_bcn_gpu_init(struct wrapper_device *device, uint32_t family)
{
   simple_mtx_lock(&device->bcn_gpu_mutex);
   if (device->bcn_tool_state != 0) {
      simple_mtx_unlock(&device->bcn_gpu_mutex);
      return;
   }
   device->bcn_tool_state = -1;
   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct bcn_pc),
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   VkResult layout = VK_ERROR_FEATURE_NOT_PRESENT, dec = layout, enc = layout, astc = layout;
   if (device->bcn_bda)
      layout = device->dispatch_table.CreatePipelineLayout(device->dispatch_handle, &plci,
                                                           NULL, &device->bcn_tool_layout);
   if (layout == VK_SUCCESS) {
      device->bcn_tool_state = 1;
      dec = bcn_create_pipeline(device, wrapper_bcn_decode_spv, sizeof(wrapper_bcn_decode_spv),
                                &device->bcn_decode_module, &device->bcn_decode_pipeline);
      enc = bcn_create_pipeline(device, wrapper_bcn_encode_spv, sizeof(wrapper_bcn_encode_spv),
                                &device->bcn_encode_module, &device->bcn_encode_pipeline);
      for (uint32_t i = 0; i < 3; i++) {
         VkResult r = bcn_create_pipeline_spec(device, wrapper_astc_encode_spv,
            sizeof(wrapper_astc_encode_spv), &device->bcn_astc_module,
            &device->bcn_astc_pipeline[i], 4 + 2 * i);
         if (i == 0 || r != VK_SUCCESS)
            astc = r;
      }
   }
   char report[160] = "self-test not run";
   if (layout == VK_SUCCESS && family != UINT32_MAX)
      bcn_selftest(device, family, report, sizeof(report));
   WRAPPER_LOG(info, "BCn GPU shaders: bda=%d layout=%d decode=%d bc-encode=%d astc-encode=%d %s",
               device->bcn_bda, layout, dec, enc, astc, report);
   wrapper_diag_append("[BCN] GPU shaders (VkResult): bda=%d layout=%d decode=%d bc-encode=%d "
                       "astc-encode=%d, %s, upload=%s\n", device->bcn_bda, layout, dec, enc,
                       astc, report,
                       wrapper_bcn_gpu_path() && device->bcn_decode_pipeline &&
                       device->bcn_astc_pipeline[0] ? "gpu" : "cpu");
   simple_mtx_unlock(&device->bcn_gpu_mutex);
}

static bool
bcn_tools_ready(struct wrapper_device *device)
{
   wrapper_bcn_gpu_init(device, UINT32_MAX);
   return device->bcn_tool_state == 1;
}

void
wrapper_bcn_gpu_finish(struct wrapper_device *device)
{
   VkDevice d = device->dispatch_handle;
   if (device->bcn_decode_pipeline)
      device->dispatch_table.DestroyPipeline(d, device->bcn_decode_pipeline, NULL);
   if (device->bcn_encode_pipeline)
      device->dispatch_table.DestroyPipeline(d, device->bcn_encode_pipeline, NULL);
   for (int i = 0; i < 3; i++)
      if (device->bcn_astc_pipeline[i])
         device->dispatch_table.DestroyPipeline(d, device->bcn_astc_pipeline[i], NULL);
   if (device->bcn_astc_module)
      device->dispatch_table.DestroyShaderModule(d, device->bcn_astc_module, NULL);
   if (device->bcn_decode_module)
      device->dispatch_table.DestroyShaderModule(d, device->bcn_decode_module, NULL);
   if (device->bcn_encode_module)
      device->dispatch_table.DestroyShaderModule(d, device->bcn_encode_module, NULL);
   if (device->bcn_tool_layout)
      device->dispatch_table.DestroyPipelineLayout(d, device->bcn_tool_layout, NULL);
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
static void
bcn_dispatch(struct wrapper_command_buffer *wcb, VkPipeline pipeline, const struct bcn_pc *pc,
             uint32_t layers)
{
   struct wrapper_device *device = wcb->device;
   bcn_barrier(wcb);
   device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
      VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, device->bcn_tool_layout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*pc), pc);
   device->dispatch_table.CmdDispatch(wcb->dispatch_handle, (pc->blocks_x + 7) / 8,
      (pc->blocks_y + 7) / 8, layers);
   bcn_barrier(wcb);
   if (wcb->compute_pipeline)
      device->dispatch_table.CmdBindPipeline(wcb->dispatch_handle,
         VK_PIPELINE_BIND_POINT_COMPUTE, wcb->compute_pipeline);
   for (uint32_t i = 0; i < wcb->push_count; i++)
      device->dispatch_table.CmdPushConstants(wcb->dispatch_handle, wcb->push[i].layout,
         wcb->push[i].stages, wcb->push[i].offset, wcb->push[i].size, wcb->push[i].data);
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
       !bcn_tools_ready(device) || !device->bcn_encode_pipeline)
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
      .flags = wi->info.format == VK_FORMAT_BC5_UNORM_BLOCK && is_astc(stored) ? 1 : 0,
      .width = ext.width,
      .height = ext.height,
   };
   bcn_dispatch(wcb, device->bcn_encode_pipeline, &pc, layers);
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
                      VkBuffer src, VkImageLayout layout, const VkBufferImageCopy *c);

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
   if (id < 0 || id == 8 || id == 9 || dst->info.imageType != VK_IMAGE_TYPE_2D ||
       !bcn_tools_ready(device) || !device->bcn_decode_pipeline) {
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
   if (!bcn_gpu_upload_region(wcb, dst, blocks->dispatch_handle, dst_layout, &up)) {
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

/* GPU upload of one region: the blocks are copied out of the application's
 * buffer, decoded, and either copied in as decoded texels or encoded to ASTC
 * first. A region that does not cover whole ASTC blocks is merged into the
 * texels already in the image: the covering blocks are unpacked, overwritten
 * and encoded again. */
static bool
bcn_gpu_upload_region(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                      VkBuffer src, VkImageLayout layout, const VkBufferImageCopy *c)
{
   struct wrapper_device *device = wcb->device;
   VkFormat format = wi->info.format;
   VkFormat stored = get_format_for_bcn(format);
   int id = bcn_format_id(format);
   uint32_t astc = bcn_astc_block(stored);
   bool decoded = stored == get_decode_format_for_bcn(format);
   if (id < 0 || id == 8 || id == 9 || (!astc && !decoded) ||
       wi->info.imageType != VK_IMAGE_TYPE_2D)
      return false;
   uint32_t w = c->imageExtent.width, h = c->imageExtent.height;
   uint32_t m = c->imageSubresource.mipLevel;
   uint32_t lw = MAX2(1u, wi->info.extent.width >> m), lh = MAX2(1u, wi->info.extent.height >> m);
   if (!w || !h || !bcn_tools_ready(device) || !device->bcn_decode_pipeline ||
       (astc && !device->bcn_astc_pipeline[astc == 4 ? 0 : astc == 6 ? 1 : 2]))
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
   uint32_t ts = astc ? 4 : get_texel_size_for_format(stored);
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
      .flags = astc ? 1 : 0,
   };
   bcn_dispatch(wcb, device->bcn_decode_pipeline, &pc, layers);

   VkBufferImageCopy copy = {
      .bufferRowLength = nbx * 4,
      .bufferImageHeight = nby * 4,
      .imageSubresource = c->imageSubresource,
      .imageOffset = c->imageOffset,
      .imageExtent = c->imageExtent,
   };
   VkBuffer from = texels->dispatch_handle;
   VkDeviceAddress enc_addr = texels_addr;
   uint32_t enc_row = nbx * 4 * 4, enc_slice = nbx * 4 * nby * 4 * 4;
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
      bcn_dispatch(wcb, device->bcn_astc_pipeline[astc == 4 ? 0 : astc == 6 ? 1 : 2], &ac,
                   layers);
      copy.bufferRowLength = abx * astc;
      copy.bufferImageHeight = aby * astc;
      from = out->dispatch_handle;
   }
   device->dispatch_table.CmdCopyBufferToImage(wcb->dispatch_handle, from, wi->dispatch_handle,
                                               layout, 1, &copy);
   return true;
}

uint32_t
wrapper_bcn_gpu_upload(struct wrapper_command_buffer *wcb, struct wrapper_image *wi,
                       VkBuffer src, VkImageLayout layout, uint32_t count,
                       const VkBufferImageCopy *regions)
{
   uint32_t done = 0;
   while (done < count && bcn_gpu_upload_region(wcb, wi, src, layout, &regions[done]))
      done++;
   return done;
}
