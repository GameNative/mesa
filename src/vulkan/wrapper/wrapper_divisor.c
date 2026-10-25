#include "wrapper_private.h"
#include "wrapper_log.h"

#include "util/hash_table.h"
#include "vk_util.h"

#define WRAPPER_DIVISOR_MAX_BINDINGS 32

struct wrapper_divisor_pipeline {
   uint32_t mask;
   bool dynamic_stride;
   uint32_t divisor[WRAPPER_DIVISOR_MAX_BINDINGS];
   uint32_t stride[WRAPPER_DIVISOR_MAX_BINDINGS];
};

struct wrapper_divisor_lowering {
   VkPipelineVertexInputStateCreateInfo vi;
   VkVertexInputBindingDescription bindings[WRAPPER_DIVISOR_MAX_BINDINGS];
   struct wrapper_divisor_pipeline info;
};

static const VkPipelineVertexInputDivisorStateCreateInfoKHR *
find_divisor_state(const VkPipelineVertexInputStateCreateInfo *vi)
{
   return vk_find_struct_const(vi->pNext, PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR);
}

static bool
has_dynamic(const VkGraphicsPipelineCreateInfo *info, VkDynamicState state)
{
   if (!info->pDynamicState)
      return false;
   for (uint32_t i = 0; i < info->pDynamicState->dynamicStateCount; i++)
      if (info->pDynamicState->pDynamicStates[i] == state)
         return true;
   return false;
}

void *
wrapper_divisor_lower(const struct wrapper_device *device,
                      const VkGraphicsPipelineCreateInfo *in,
                      VkGraphicsPipelineCreateInfo *out)
{
   *out = *in;
   if (!device->emulate_vertex_divisor || !in->pVertexInputState ||
       has_dynamic(in, VK_DYNAMIC_STATE_VERTEX_INPUT_EXT))
      return NULL;

   const VkPipelineVertexInputStateCreateInfo *vi = in->pVertexInputState;
   const VkPipelineVertexInputDivisorStateCreateInfoKHR *ds = find_divisor_state(vi);
   if (!ds || vi->vertexBindingDescriptionCount > WRAPPER_DIVISOR_MAX_BINDINGS)
      return NULL;

   struct wrapper_divisor_lowering *l = calloc(1, sizeof(*l));
   if (!l)
      return NULL;
   l->vi = *vi;
   memcpy(l->bindings, vi->pVertexBindingDescriptions,
          vi->vertexBindingDescriptionCount * sizeof(*l->bindings));
   l->vi.pVertexBindingDescriptions = l->bindings;
   if (vi->pNext == (const void *)ds)
      l->vi.pNext = ds->pNext;
   l->info.dynamic_stride = has_dynamic(in, VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);

   for (uint32_t i = 0; i < ds->vertexBindingDivisorCount; i++) {
      const VkVertexInputBindingDivisorDescriptionKHR *d = &ds->pVertexBindingDivisors[i];
      if (d->divisor == 1 || d->binding >= WRAPPER_DIVISOR_MAX_BINDINGS)
         continue;
      for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++) {
         if (l->bindings[b].binding != d->binding ||
             l->bindings[b].inputRate != VK_VERTEX_INPUT_RATE_INSTANCE)
            continue;
         l->info.mask |= 1u << d->binding;
         l->info.divisor[d->binding] = d->divisor;
         l->info.stride[d->binding] = l->bindings[b].stride;
         l->bindings[b].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
         l->bindings[b].stride = 0;
      }
   }
   out->pVertexInputState = &l->vi;
   return l;
}

void
wrapper_divisor_record(struct wrapper_device *device, void *lowering, VkPipeline pipeline)
{
   struct wrapper_divisor_lowering *l = lowering;
   if (!l || pipeline == VK_NULL_HANDLE || !l->info.mask)
      return;
   struct wrapper_divisor_pipeline *p = malloc(sizeof(*p));
   if (!p)
      return;
   *p = l->info;
   simple_mtx_lock(&device->resource_mutex);
   _mesa_hash_table_u64_insert(device->divisor_pipeline_table, (uint64_t)pipeline, p);
   simple_mtx_unlock(&device->resource_mutex);
}

void
wrapper_divisor_destroy(struct wrapper_device *device, VkPipeline pipeline)
{
   simple_mtx_lock(&device->resource_mutex);
   struct wrapper_divisor_pipeline *p =
      _mesa_hash_table_u64_search(device->divisor_pipeline_table, (uint64_t)pipeline);
   if (p)
      _mesa_hash_table_u64_remove(device->divisor_pipeline_table, (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);
   free(p);
}

void
wrapper_divisor_bind(struct wrapper_command_buffer *wcb, VkPipeline pipeline)
{
   struct wrapper_device *device = wcb->device;
   simple_mtx_lock(&device->resource_mutex);
   wcb->divisor_pipeline =
      _mesa_hash_table_u64_search(device->divisor_pipeline_table, (uint64_t)pipeline);
   simple_mtx_unlock(&device->resource_mutex);
}

void
wrapper_divisor_track(struct wrapper_command_buffer *wcb, uint32_t first, uint32_t count,
                      const VkBuffer *buffers, const VkDeviceSize *offsets,
                      const VkDeviceSize *sizes, const VkDeviceSize *strides)
{
   for (uint32_t i = 0; i < count && first + i < WRAPPER_DIVISOR_MAX_BINDINGS; i++) {
      uint32_t b = first + i;
      wcb->vb_buffer[b] = buffers[i];
      wcb->vb_offset[b] = offsets[i];
      wcb->vb_size[b] = sizes ? sizes[i] : VK_WHOLE_SIZE;
      if (strides)
         wcb->vb_stride[b] = strides[i];
      wcb->vb_valid |= 1u << b;
   }
}

void
wrapper_divisor_reset(struct wrapper_command_buffer *wcb)
{
   wcb->divisor_pipeline = NULL;
   wcb->vb_valid = 0;
}

void
wrapper_divisor_finish(struct wrapper_device *device)
{
   hash_table_u64_foreach(device->divisor_pipeline_table, entry)
      free(entry.data);
   _mesa_hash_table_u64_destroy(device->divisor_pipeline_table);
   device->divisor_pipeline_table = NULL;
}

static uint32_t
gcd(uint32_t a, uint32_t b)
{
   while (b) {
      uint32_t t = a % b;
      a = b;
      b = t;
   }
   return a;
}

static void
bind_one(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
         uint32_t b, VkDeviceSize offset, VkDeviceSize size, VkDeviceSize stride)
{
   struct wrapper_device *device = wcb->device;
   VkBuffer buffer = wcb->vb_buffer[b];
   if (p->dynamic_stride && !device->emulate_dynamic_stride &&
       device->dispatch_table.CmdBindVertexBuffers2) {
      device->dispatch_table.CmdBindVertexBuffers2(wcb->dispatch_handle, b, 1, &buffer,
                                                   &offset, &size, &stride);
   } else {
      device->dispatch_table.CmdBindVertexBuffers(wcb->dispatch_handle, b, 1, &buffer,
                                                  &offset);
   }
}

static VkDeviceSize
binding_stride(const struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
               uint32_t b)
{
   return p->dynamic_stride ? wcb->vb_stride[b] : p->stride[b];
}

static void
override_strides(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
                 uint32_t *saved, uint32_t *saved_valid)
{
   *saved_valid = wcb->vertex_strides_valid;
   u_foreach_bit(b, p->mask) {
      saved[b] = wcb->vertex_strides[b];
      wcb->vertex_strides[b] = 0;
      wcb->vertex_strides_valid |= 1u << b;
   }
}

static void
restore_strides(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
                const uint32_t *saved, uint32_t saved_valid)
{
   u_foreach_bit(b, p->mask)
      wcb->vertex_strides[b] = saved[b];
   wcb->vertex_strides_valid = saved_valid;
}

static void
bind_elements(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p,
              uint32_t first_instance, uint32_t start)
{
   u_foreach_bit(b, p->mask & wcb->vb_valid) {
      uint32_t d = p->divisor[b];
      VkDeviceSize stride = binding_stride(wcb, p, b);
      VkDeviceSize element = d ? first_instance + start / d : first_instance;
      VkDeviceSize skip = element * stride;
      VkDeviceSize size = wcb->vb_size[b];
      if (size != VK_WHOLE_SIZE)
         size = size > skip ? size - skip : 0;
      bind_one(wcb, p, b, wcb->vb_offset[b] + skip, size, 0);
   }
}

static void
restore_bindings(struct wrapper_command_buffer *wcb, const struct wrapper_divisor_pipeline *p)
{
   u_foreach_bit(b, p->mask & wcb->vb_valid)
      bind_one(wcb, p, b, wcb->vb_offset[b], wcb->vb_size[b], binding_stride(wcb, p, b));
}

bool
wrapper_divisor_draw(struct wrapper_command_buffer *wcb, bool indexed, uint32_t count,
                     uint32_t instance_count, uint32_t first, int32_t vertex_offset,
                     uint32_t first_instance)
{
   const struct wrapper_divisor_pipeline *p = wcb->divisor_pipeline;
   struct wrapper_device *device = wcb->device;
   if (!p || !(p->mask & wcb->vb_valid))
      return false;

   uint32_t group = 0;
   u_foreach_bit(b, p->mask)
      if (p->divisor[b])
         group = group ? gcd(group, p->divisor[b]) : p->divisor[b];
   if (!group)
      group = MAX2(instance_count, 1);

   uint32_t saved[WRAPPER_DIVISOR_MAX_BINDINGS], saved_valid = 0;
   if (wcb->stride_pipeline) {
      override_strides(wcb, p, saved, &saved_valid);
      wrapper_stride_flush(wcb);
   }

   for (uint32_t start = 0; start < instance_count; start += group) {
      uint32_t n = MIN2(group, instance_count - start);
      bind_elements(wcb, p, first_instance, start);
      if (indexed)
         device->dispatch_table.CmdDrawIndexed(wcb->dispatch_handle, count, n, first,
                                               vertex_offset, first_instance + start);
      else
         device->dispatch_table.CmdDraw(wcb->dispatch_handle, count, n, first,
                                        first_instance + start);
   }

   restore_bindings(wcb, p);
   if (wcb->stride_pipeline)
      restore_strides(wcb, p, saved, saved_valid);
   return true;
}

bool
wrapper_divisor_begin_indirect(struct wrapper_command_buffer *wcb)
{
   const struct wrapper_divisor_pipeline *p = wcb->divisor_pipeline;
   if (!p || !(p->mask & wcb->vb_valid))
      return false;
   static bool warned;
   u_foreach_bit(b, p->mask) {
      if (p->divisor[b] > 1 && !warned) {
         warned = true;
         WRAPPER_LOG(error, "Indirect draw with vertex attribute divisor %u: "
                            "every instance reads the first element", p->divisor[b]);
      }
   }
   if (wcb->stride_pipeline) {
      override_strides(wcb, p, wcb->divisor_saved_strides, &wcb->divisor_saved_valid);
      wrapper_stride_flush(wcb);
   }
   bind_elements(wcb, p, 0, 0);
   return true;
}

void
wrapper_divisor_end_indirect(struct wrapper_command_buffer *wcb)
{
   const struct wrapper_divisor_pipeline *p = wcb->divisor_pipeline;
   restore_bindings(wcb, p);
   if (wcb->stride_pipeline)
      restore_strides(wcb, p, wcb->divisor_saved_strides, wcb->divisor_saved_valid);
}
