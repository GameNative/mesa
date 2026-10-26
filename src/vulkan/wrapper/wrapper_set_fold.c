#include <string.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "vk_util.h"
#include "util/hash_table.h"
#include "util/u_math.h"
#include "util/u_debug.h"

#define FOLD_MAX_MEMBERS 8
#define FOLD_MAX_BINDINGS 16

struct fold_binding {
   VkDescriptorSetLayoutBinding vk;
   VkDescriptorBindingFlags flags;
};

struct fold_layout {
   VkDescriptorSetLayoutCreateFlags flags;
   uint32_t binding_count;
   struct fold_binding bindings[FOLD_MAX_BINDINGS];
   uint32_t variable_binding;
   uint32_t top;
};

struct fold_merge {
   struct fold_merge *next;
   uint32_t member_count;
   VkDescriptorSetLayout handles[FOLD_MAX_MEMBERS];
   struct fold_layout *members[FOLD_MAX_MEMBERS];
   uint32_t base[FOLD_MAX_MEMBERS];
   VkDescriptorSetLayout merged;
   uint32_t binding_count;
   VkDescriptorSetLayoutBinding bindings[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   VkDescriptorBindingFlags binding_flags[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   uint32_t declared_variable;
};

struct fold_instance {
   struct fold_merge *merge;
   uint32_t live;
   VkDescriptorSet sets[FOLD_MAX_MEMBERS];
   uint32_t variable_count;
   VkDescriptorPool pool;
   VkDescriptorSet set;
   bool truncation_logged;
};

struct fold_set {
   VkDescriptorPool pool;
   VkDescriptorSetLayout layout_handle;
   struct fold_layout *layout;
   uint32_t variable_count;
   struct fold_instance *instance;
   uint32_t member;
};

struct fold_pipeline_layout {
   uint32_t run_start;
   uint32_t run_len;
   struct fold_merge *merge;
};

static bool
is_view_type(VkDescriptorType t)
{
   return t != VK_DESCRIPTOR_TYPE_SAMPLER &&
          t != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
          t != VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          t != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK &&
          t != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
          t != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

void
wrapper_fold_init(struct wrapper_device *device)
{
   simple_mtx_init(&device->fold_mutex, mtx_plain);
   device->fold_layout_table = _mesa_hash_table_u64_create(NULL);
   device->fold_pl_table = _mesa_hash_table_u64_create(NULL);
   device->fold_set_table = _mesa_hash_table_u64_create(NULL);
   device->fold_set_budget = debug_get_num_option("WRAPPER_FOLD_SET_BUDGET", 65536);
   uint32_t v = device->physical->properties2.properties.driverVersion;
   WRAPPER_LOG(info, "Folded descriptor set budget %u descriptors (driver %u.%u.%u); "
               "descriptors at a set offset above it read as zero on Mali r26",
               device->fold_set_budget, VK_VERSION_MAJOR(v), VK_VERSION_MINOR(v),
               VK_VERSION_PATCH(v));
}

void
wrapper_fold_note_pool(struct wrapper_device *device, const VkDescriptorPoolCreateInfo *info)
{
   if (!(info->flags & VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT))
      return;
   for (uint32_t i = 0; i < info->poolSizeCount; i++)
      if (is_view_type(info->pPoolSizes[i].type))
         device->fold_max_pool_count =
            MAX2(device->fold_max_pool_count, info->pPoolSizes[i].descriptorCount);
}

void
wrapper_fold_note_layout(struct wrapper_device *device,
                         const VkDescriptorSetLayoutCreateInfo *info,
                         VkDescriptorSetLayout handle)
{
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *flags_info =
      vk_find_struct_const(info->pNext, DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   int variable = -1;

   if (!flags_info || info->bindingCount > FOLD_MAX_BINDINGS)
      return;
   for (uint32_t i = 0; i < flags_info->bindingCount && i < info->bindingCount; i++)
      if (flags_info->pBindingFlags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         variable = i;
   if (variable < 0 || !is_view_type(info->pBindings[variable].descriptorType))
      return;

   struct fold_layout *l = calloc(1, sizeof(*l));
   if (!l)
      return;
   l->flags = info->flags;
   l->binding_count = info->bindingCount;
   l->variable_binding = variable;
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      l->bindings[i].vk = info->pBindings[i];
      l->bindings[i].vk.pImmutableSamplers = NULL;
      l->bindings[i].flags = i < flags_info->bindingCount ? flags_info->pBindingFlags[i] : 0;
      l->top = MAX2(l->top, info->pBindings[i].binding + 1);
   }
   simple_mtx_lock(&device->fold_mutex);
   _mesa_hash_table_u64_insert(device->fold_layout_table, (uint64_t)handle, l);
   simple_mtx_unlock(&device->fold_mutex);
}

static struct fold_layout *
fold_layout_lookup(struct wrapper_device *device, VkDescriptorSetLayout handle)
{
   return _mesa_hash_table_u64_search(device->fold_layout_table, (uint64_t)handle);
}

enum fold_class {
   FOLD_CLASS_SAMPLER,
   FOLD_CLASS_SAMPLED,
   FOLD_CLASS_STORAGE_IMAGE,
   FOLD_CLASS_UBO,
   FOLD_CLASS_SSBO,
   FOLD_CLASS_COUNT,
};

static uint32_t
fold_class_mask(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      return BITFIELD_BIT(FOLD_CLASS_SAMPLER);
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return BITFIELD_BIT(FOLD_CLASS_SAMPLER) | BITFIELD_BIT(FOLD_CLASS_SAMPLED);
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      return BITFIELD_BIT(FOLD_CLASS_SAMPLED);
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      return BITFIELD_BIT(FOLD_CLASS_STORAGE_IMAGE);
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      return BITFIELD_BIT(FOLD_CLASS_UBO);
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      return BITFIELD_BIT(FOLD_CLASS_SSBO);
   default:
      return 0;
   }
}

static uint32_t
fold_weight(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return 4;
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return 3;
   default:
      return 1;
   }
}

static uint32_t
fold_binding_cap(const uint32_t *class_cap, uint32_t resource_cap, VkDescriptorType type)
{
   uint32_t cap = resource_cap;
   uint32_t mask = fold_class_mask(type);
   for (uint32_t c = 0; c < FOLD_CLASS_COUNT; c++)
      if (mask & BITFIELD_BIT(c))
         cap = MIN2(cap, class_cap[c]);
   return MAX2(cap, 1);
}

static struct fold_merge *
fold_merge_get(struct wrapper_device *device, const VkDescriptorSetLayout *handles, uint32_t count)
{
   for (struct fold_merge *m = device->fold_merges; m; m = m->next)
      if (m->member_count == count && !memcmp(m->handles, handles, count * sizeof(*handles)))
         return m->merged ? m : NULL;

   struct fold_merge *m = calloc(1, sizeof(*m));
   if (!m)
      return NULL;
   m->member_count = count;
   memcpy(m->handles, handles, count * sizeof(*handles));
   m->next = device->fold_merges;
   device->fold_merges = m;

   const struct vk_properties *p = &device->physical->vk.properties;
   const uint32_t limits[FOLD_CLASS_COUNT] = {
      [FOLD_CLASS_SAMPLER] = MIN2(p->maxPerStageDescriptorUpdateAfterBindSamplers,
                                  p->maxDescriptorSetUpdateAfterBindSamplers),
      [FOLD_CLASS_SAMPLED] = MIN2(p->maxPerStageDescriptorUpdateAfterBindSampledImages,
                                  p->maxDescriptorSetUpdateAfterBindSampledImages),
      [FOLD_CLASS_STORAGE_IMAGE] = MIN2(p->maxPerStageDescriptorUpdateAfterBindStorageImages,
                                        p->maxDescriptorSetUpdateAfterBindStorageImages),
      [FOLD_CLASS_UBO] = MIN2(p->maxPerStageDescriptorUpdateAfterBindUniformBuffers,
                              p->maxDescriptorSetUpdateAfterBindUniformBuffers),
      [FOLD_CLASS_SSBO] = MIN2(p->maxPerStageDescriptorUpdateAfterBindStorageBuffers,
                               p->maxDescriptorSetUpdateAfterBindStorageBuffers),
   };
   uint64_t small[FOLD_CLASS_COUNT] = {0}, small_total = 0;
   uint32_t arrays[FOLD_CLASS_COUNT] = {0};
   for (uint32_t k = 0; k < count; k++) {
      struct fold_layout *l = fold_layout_lookup(device, handles[k]);
      for (uint32_t b = 0; b < l->binding_count; b++) {
         const VkDescriptorSetLayoutBinding *vb = &l->bindings[b].vk;
         uint32_t mask = fold_class_mask(vb->descriptorType);
         for (uint32_t c = 0; c < FOLD_CLASS_COUNT; c++) {
            if (!(mask & BITFIELD_BIT(c)))
               continue;
            if (b == l->variable_binding)
               arrays[c]++;
            else
               small[c] += vb->descriptorCount;
         }
         if (b != l->variable_binding)
            small_total += vb->descriptorCount;
      }
   }
   uint64_t resources = p->maxPerStageUpdateAfterBindResources;
   uint32_t resource_cap = resources > small_total ? (resources - small_total) / count : 1;
   uint64_t budget = device->fold_set_budget;
   uint64_t room = budget > small_total ? budget - small_total : count;
   uint32_t weight_sum = 0;
   for (uint32_t k = 0; k < count; k++) {
      struct fold_layout *l = fold_layout_lookup(device, handles[k]);
      if (l->variable_binding < l->binding_count)
         weight_sum += fold_weight(l->bindings[l->variable_binding].vk.descriptorType);
   }
   uint32_t class_cap[FOLD_CLASS_COUNT];
   for (uint32_t c = 0; c < FOLD_CLASS_COUNT; c++) {
      uint64_t room = limits[c] > small[c] ? limits[c] - small[c] : 0;
      class_cap[c] = arrays[c] ? MAX2(room / arrays[c], 1) : UINT32_MAX;
   }
   uint32_t override = debug_get_num_option("WRAPPER_FOLD_ARRAY_SIZE", 0);
   static const char *const class_env[FOLD_CLASS_COUNT] = {
      [FOLD_CLASS_SAMPLER] = "WRAPPER_FOLD_SIZE_SAMPLER",
      [FOLD_CLASS_SAMPLED] = "WRAPPER_FOLD_SIZE_SAMPLED",
      [FOLD_CLASS_STORAGE_IMAGE] = "WRAPPER_FOLD_SIZE_STORAGE_IMAGE",
      [FOLD_CLASS_UBO] = "WRAPPER_FOLD_SIZE_UBO",
      [FOLD_CLASS_SSBO] = "WRAPPER_FOLD_SIZE_SSBO",
   };
   for (uint32_t c = 0; c < FOLD_CLASS_COUNT; c++) {
      uint32_t v = debug_get_num_option(class_env[c], 0);
      if (v)
         class_cap[c] = MIN2(class_cap[c], v);
      else if (override)
         class_cap[c] = MIN2(class_cap[c], override);
   }
   VkDescriptorSetLayoutCreateFlags flags = 0;
   uint32_t n = 0, base = 0;
   uint64_t total = 0;

   for (uint32_t k = 0; k < count; k++) {
      struct fold_layout *l = fold_layout_lookup(device, handles[k]);
      bool last = k == count - 1;
      m->members[k] = l;
      m->base[k] = base;
      flags |= l->flags;
      for (uint32_t b = 0; b < l->binding_count; b++) {
         VkDescriptorSetLayoutBinding vb = l->bindings[b].vk;
         VkDescriptorBindingFlags bf = l->bindings[b].flags;
         vb.binding += base;
         if (b == l->variable_binding) {
            uint32_t share = (uint32_t)MAX2(room * fold_weight(vb.descriptorType) /
                                            MAX2(weight_sum, 1), 1);
            uint32_t cap = fold_binding_cap(class_cap, MIN2(resource_cap, share),
                                            vb.descriptorType);
            vb.descriptorCount = MIN2(vb.descriptorCount, cap);
            if (last) {
               m->declared_variable = vb.descriptorCount;
            } else {
               bf &= ~VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
               total += vb.descriptorCount;
            }
            WRAPPER_LOG(info, "Folded set binding %u (type %d): cap %u", vb.binding,
                        vb.descriptorType, vb.descriptorCount);
         } else {
            total += vb.descriptorCount;
         }
         m->bindings[n] = vb;
         m->binding_flags[n] = bf;
         n++;
      }
      base += l->top;
   }
   m->binding_count = n;

   VkDescriptorSetLayoutBindingFlagsCreateInfo bfi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = n,
      .pBindingFlags = m->binding_flags,
   };
   VkDescriptorSetLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &bfi,
      .flags = flags,
      .bindingCount = n,
      .pBindings = m->bindings,
   };
   VkResult r = device->dispatch_table.CreateDescriptorSetLayout(device->dispatch_handle,
                                                                 &ci, NULL, &m->merged);
   if (r != VK_SUCCESS)
      m->merged = VK_NULL_HANDLE;
   WRAPPER_LOG(info, "Folded %u bindless set layouts into one: %u bindings, set budget %u, "
               "%llu fixed descriptors, result %d",
               count, n, device->fold_set_budget, (unsigned long long)total, r);
   return m->merged ? m : NULL;
}

bool
wrapper_fold_create_pipeline_layout(struct wrapper_device *device,
                                    const VkPipelineLayoutCreateInfo *info,
                                    const VkAllocationCallbacks *alloc,
                                    VkPipelineLayout *out, VkResult *result)
{
   uint32_t max_sets = device->physical->properties2.properties.limits.maxBoundDescriptorSets;
   if (info->setLayoutCount <= max_sets)
      return false;

   uint32_t best_start = 0, best_len = 0;
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < info->setLayoutCount;) {
      uint32_t len = 0;
      while (i + len < info->setLayoutCount && len < FOLD_MAX_MEMBERS &&
             fold_layout_lookup(device, info->pSetLayouts[i + len]))
         len++;
      if (len > best_len) {
         best_len = len;
         best_start = i;
      }
      i += len ? len : 1;
   }
   struct fold_merge *merge = NULL;
   if (best_len >= 2 && info->setLayoutCount - best_len + 1 <= max_sets)
      merge = fold_merge_get(device, &info->pSetLayouts[best_start], best_len);
   simple_mtx_unlock(&device->fold_mutex);

   if (!merge) {
      WRAPPER_LOG(error, "Pipeline layout needs %u sets, driver has %u, cannot fold (run %u)",
                  info->setLayoutCount, max_sets, best_len);
      *result = VK_ERROR_INITIALIZATION_FAILED;
      return true;
   }

   uint32_t count = info->setLayoutCount - best_len + 1;
   VkDescriptorSetLayout sets[32];
   if (count > ARRAY_SIZE(sets)) {
      *result = VK_ERROR_INITIALIZATION_FAILED;
      return true;
   }
   for (uint32_t i = 0, o = 0; i < info->setLayoutCount; i++) {
      if (i == best_start)
         sets[o++] = merge->merged;
      else if (i < best_start || i >= best_start + best_len)
         sets[o++] = info->pSetLayouts[i];
   }
   VkPipelineLayoutCreateInfo ci = *info;
   ci.setLayoutCount = count;
   ci.pSetLayouts = sets;
   *result = device->dispatch_table.CreatePipelineLayout(device->dispatch_handle, &ci, alloc, out);
   if (*result != VK_SUCCESS)
      return true;

   struct fold_pipeline_layout *pl = calloc(1, sizeof(*pl));
   if (pl) {
      pl->run_start = best_start;
      pl->run_len = best_len;
      pl->merge = merge;
      simple_mtx_lock(&device->fold_mutex);
      _mesa_hash_table_u64_insert(device->fold_pl_table, (uint64_t)*out, pl);
      simple_mtx_unlock(&device->fold_mutex);
   }
   WRAPPER_LOG(info, "Folding active for pipeline layout %u sets -> %u (sets %u-%u merged)",
               info->setLayoutCount, count, best_start, best_start + best_len - 1);
   return true;
}

void
wrapper_fold_destroy_pipeline_layout(struct wrapper_device *device, VkPipelineLayout layout)
{
   simple_mtx_lock(&device->fold_mutex);
   struct fold_pipeline_layout *pl =
      _mesa_hash_table_u64_search(device->fold_pl_table, (uint64_t)layout);
   if (pl)
      _mesa_hash_table_u64_remove(device->fold_pl_table, (uint64_t)layout);
   simple_mtx_unlock(&device->fold_mutex);
   free(pl);
}

static struct fold_pipeline_layout *
fold_pl_lookup(struct wrapper_device *device, VkPipelineLayout layout)
{
   simple_mtx_lock(&device->fold_mutex);
   struct fold_pipeline_layout *pl =
      _mesa_hash_table_u64_search(device->fold_pl_table, (uint64_t)layout);
   simple_mtx_unlock(&device->fold_mutex);
   return pl;
}

bool
wrapper_fold_layout_folded(struct wrapper_device *device, VkPipelineLayout layout)
{
   return fold_pl_lookup(device, layout) != NULL;
}

static uint32_t
fold_index(const struct fold_pipeline_layout *pl, uint32_t set)
{
   if (set < pl->run_start)
      return set;
   if (set < pl->run_start + pl->run_len)
      return pl->run_start;
   return set - pl->run_len + 1;
}

uint32_t
wrapper_fold_set_index(struct wrapper_device *device, VkPipelineLayout layout, uint32_t set)
{
   struct fold_pipeline_layout *pl = fold_pl_lookup(device, layout);
   return pl ? fold_index(pl, set) : set;
}

uint32_t *
wrapper_fold_remap_spirv(struct wrapper_device *device, VkPipelineLayout layout,
                         const uint32_t *code, size_t words)
{
   struct fold_pipeline_layout *pl = fold_pl_lookup(device, layout);
   if (!pl || words < 5)
      return NULL;
   uint32_t bound = code[3];
   uint32_t *set_of = malloc(bound * sizeof(uint32_t));
   uint32_t *out = malloc(words * sizeof(uint32_t));
   if (!set_of || !out) {
      free(set_of);
      free(out);
      return NULL;
   }
   memset(set_of, 0xff, bound * sizeof(uint32_t));
   memcpy(out, code, words * sizeof(uint32_t));

   for (size_t i = 5; i < words;) {
      uint32_t op = code[i] & 0xffff, n = code[i] >> 16;
      if (!n)
         break;
      if (op == 71 && n >= 4 && code[i + 2] == 34 && code[i + 1] < bound)
         set_of[code[i + 1]] = code[i + 3];
      i += n;
   }

   for (size_t i = 5; i < words;) {
      uint32_t op = code[i] & 0xffff, n = code[i] >> 16;
      if (!n)
         break;
      if (op == 71 && n >= 4 && code[i + 1] < bound) {
         uint32_t s = set_of[code[i + 1]];
         if (s != UINT32_MAX && s >= pl->run_start) {
            bool in_run = s < pl->run_start + pl->run_len;
            if (code[i + 2] == 34)
               out[i + 3] = fold_index(pl, s);
            else if (code[i + 2] == 33 && in_run)
               out[i + 3] = code[i + 3] + pl->merge->base[s - pl->run_start];
         }
      }
      i += n;
   }
   free(set_of);
   return out;
}

void
wrapper_fold_note_sets(struct wrapper_device *device, const VkDescriptorSetAllocateInfo *info,
                       const VkDescriptorSet *sets)
{
   const VkDescriptorSetVariableDescriptorCountAllocateInfo *vai =
      vk_find_struct_const(info->pNext, DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
      struct fold_layout *l = fold_layout_lookup(device, info->pSetLayouts[i]);
      if (!l)
         continue;
      struct fold_set *fs = calloc(1, sizeof(*fs));
      if (!fs)
         continue;
      fs->pool = info->descriptorPool;
      fs->layout_handle = info->pSetLayouts[i];
      fs->layout = l;
      fs->variable_count = vai && i < vai->descriptorSetCount ? vai->pDescriptorCounts[i] : 0;
      _mesa_hash_table_u64_insert(device->fold_set_table, (uint64_t)sets[i], fs);
   }
   simple_mtx_unlock(&device->fold_mutex);
}

static void
fold_instance_drop(struct wrapper_device *device, struct fold_instance *inst)
{
   if (--inst->live)
      return;
   if (inst->pool)
      device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, inst->pool, NULL);
   free(inst);
}

static void
fold_set_drop(struct wrapper_device *device, uint64_t key, struct fold_set *fs)
{
   _mesa_hash_table_u64_remove(device->fold_set_table, key);
   if (fs->instance)
      fold_instance_drop(device, fs->instance);
   free(fs);
}

void
wrapper_fold_free_sets(struct wrapper_device *device, uint32_t count, const VkDescriptorSet *sets)
{
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < count; i++) {
      struct fold_set *fs = sets[i]
         ? _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)sets[i]) : NULL;
      if (fs)
         fold_set_drop(device, (uint64_t)sets[i], fs);
   }
   simple_mtx_unlock(&device->fold_mutex);
}

void
wrapper_fold_reset_pool(struct wrapper_device *device, VkDescriptorPool pool)
{
   simple_mtx_lock(&device->fold_mutex);
   hash_table_u64_foreach(device->fold_set_table, entry) {
      struct fold_set *fs = entry.data;
      if (fs->pool == pool)
         fold_set_drop(device, entry.key, fs);
   }
   simple_mtx_unlock(&device->fold_mutex);
}

static uint32_t
fold_member_limit(const struct fold_instance *inst, uint32_t member, uint32_t binding)
{
   const struct fold_merge *m = inst->merge;
   const struct fold_layout *l = m->members[member];
   if (member == m->member_count - 1 && l->bindings[l->variable_binding].vk.binding == binding)
      return inst->variable_count;
   for (uint32_t i = 0; i < m->binding_count; i++)
      if (m->bindings[i].binding == binding + m->base[member])
         return m->bindings[i].descriptorCount;
   return 0;
}

static struct fold_instance *
fold_instance_create(struct wrapper_device *device, struct fold_merge *m,
                     const VkDescriptorSet *sets)
{
   struct fold_set *fsets[FOLD_MAX_MEMBERS];
   for (uint32_t k = 0; k < m->member_count; k++) {
      fsets[k] = _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)sets[k]);
      if (!fsets[k] || fsets[k]->layout_handle != m->handles[k])
         return NULL;
   }

   struct fold_instance *inst = calloc(1, sizeof(*inst));
   if (!inst)
      return NULL;
   inst->merge = m;
   inst->variable_count = MIN2(fsets[m->member_count - 1]->variable_count, m->declared_variable);

   VkDescriptorPoolSize sizes[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   uint32_t size_count = 0;
   uint64_t total = 0;
   for (uint32_t i = 0; i < m->binding_count; i++) {
      uint32_t c = (m->binding_flags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         ? inst->variable_count : m->bindings[i].descriptorCount;
      uint32_t j;
      for (j = 0; j < size_count; j++)
         if (sizes[j].type == m->bindings[i].descriptorType)
            break;
      if (j == size_count) {
         sizes[size_count].type = m->bindings[i].descriptorType;
         sizes[size_count++].descriptorCount = 0;
      }
      sizes[j].descriptorCount += MAX2(c, 1);
      total += c;
   }
   VkDescriptorPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets = 1,
      .poolSizeCount = size_count,
      .pPoolSizes = sizes,
   };
   VkDescriptorSetVariableDescriptorCountAllocateInfo vai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
      .descriptorSetCount = 1,
      .pDescriptorCounts = &inst->variable_count,
   };
   VkDescriptorSetAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = &vai,
      .descriptorSetCount = 1,
      .pSetLayouts = &m->merged,
   };
   VkResult r = device->dispatch_table.CreateDescriptorPool(device->dispatch_handle, &pci, NULL,
                                                            &inst->pool);
   if (r == VK_SUCCESS) {
      ai.descriptorPool = inst->pool;
      r = device->dispatch_table.AllocateDescriptorSets(device->dispatch_handle, &ai, &inst->set);
   }
   WRAPPER_LOG(info, "Folded descriptor set allocated: %llu descriptors (variable %u), result %d",
               (unsigned long long)total, inst->variable_count, r);
   if (r != VK_SUCCESS) {
      if (inst->pool)
         device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, inst->pool, NULL);
      free(inst);
      return NULL;
   }

   VkCopyDescriptorSet copies[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   uint32_t copy_count = 0;
   for (uint32_t k = 0; k < m->member_count; k++) {
      const struct fold_layout *l = m->members[k];
      for (uint32_t b = 0; b < l->binding_count; b++) {
         uint32_t count = l->bindings[b].vk.descriptorCount;
         if (b == l->variable_binding)
            count = fsets[k]->variable_count;
         uint32_t limit = fold_member_limit(inst, k, l->bindings[b].vk.binding);
         if (count > limit && !inst->truncation_logged) {
            inst->truncation_logged = true;
            WRAPPER_LOG(info, "Folded heap set %p: binding %u holds %u descriptors but only %u "
                        "fit under the set budget; descriptors above %u are dropped",
                        (void *)inst, l->bindings[b].vk.binding, count, limit, limit - 1);
         }
         count = MIN2(count, limit);
         if (!count)
            continue;
         copies[copy_count++] = (VkCopyDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
            .srcSet = sets[k],
            .srcBinding = l->bindings[b].vk.binding,
            .dstSet = inst->set,
            .dstBinding = l->bindings[b].vk.binding + m->base[k],
            .descriptorCount = count,
         };
      }
   }
   if (copy_count)
      device->dispatch_table.UpdateDescriptorSets(device->dispatch_handle, 0, NULL,
                                                  copy_count, copies);
   WRAPPER_LOG(info, "Folded set migrated: %u bindings copied from %u member sets",
               copy_count, m->member_count);

   for (uint32_t k = 0; k < m->member_count; k++) {
      if (fsets[k]->instance)
         fold_instance_drop(device, fsets[k]->instance);
      fsets[k]->instance = inst;
      fsets[k]->member = k;
      inst->sets[k] = sets[k];
      inst->live++;
   }
   return inst;
}

static bool
fold_resolve(struct wrapper_device *device, VkDescriptorSet *set, uint32_t *binding,
             uint32_t *element, uint32_t *count)
{
   struct fold_set *fs = _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)*set);
   if (!fs || !fs->instance)
      return true;
   uint32_t limit = fold_member_limit(fs->instance, fs->member, *binding);
   if (*element >= limit) {
      if (!fs->instance->truncation_logged) {
         fs->instance->truncation_logged = true;
         WRAPPER_LOG(info, "Folded heap set %p: descriptor %u of binding %u is above the array size %u; "
                     "accesses above it are dropped", (void *)(uintptr_t)fs->instance->set,
                     *element, *binding, limit);
      }
      return false;
   }
   *count = MIN2(*count, limit - *element);
   *binding += fs->instance->merge->base[fs->member];
   *set = fs->instance->set;
   return true;
}

bool
wrapper_fold_remap_updates(struct wrapper_device *device,
                           uint32_t *write_count, VkWriteDescriptorSet **writes,
                           uint32_t *copy_count, VkCopyDescriptorSet **copies)
{
   bool touched = false;
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < *write_count && !touched; i++) {
      struct fold_set *fs =
         _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)(*writes)[i].dstSet);
      touched = fs && fs->instance;
   }
   for (uint32_t i = 0; i < *copy_count && !touched; i++) {
      struct fold_set *a =
         _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)(*copies)[i].dstSet);
      struct fold_set *b =
         _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)(*copies)[i].srcSet);
      touched = (a && a->instance) || (b && b->instance);
   }
   if (!touched) {
      simple_mtx_unlock(&device->fold_mutex);
      return false;
   }

   VkWriteDescriptorSet *w = malloc(MAX2(*write_count, 1) * sizeof(*w));
   VkCopyDescriptorSet *c = malloc(MAX2(*copy_count, 1) * sizeof(*c));
   uint32_t wn = 0, cn = 0;
   if (w && c) {
      for (uint32_t i = 0; i < *write_count; i++) {
         VkWriteDescriptorSet x = (*writes)[i];
         if (fold_resolve(device, &x.dstSet, &x.dstBinding, &x.dstArrayElement,
                          &x.descriptorCount) && x.descriptorCount)
            w[wn++] = x;
      }
      for (uint32_t i = 0; i < *copy_count; i++) {
         VkCopyDescriptorSet x = (*copies)[i];
         if (fold_resolve(device, &x.srcSet, &x.srcBinding, &x.srcArrayElement,
                          &x.descriptorCount) &&
             fold_resolve(device, &x.dstSet, &x.dstBinding, &x.dstArrayElement,
                          &x.descriptorCount) && x.descriptorCount)
            c[cn++] = x;
      }
   }
   simple_mtx_unlock(&device->fold_mutex);
   *writes = w;
   *copies = c;
   *write_count = wn;
   *copy_count = cn;
   return true;
}

static struct fold_instance *
fold_instance_for_member(struct wrapper_device *device, struct fold_merge *m,
                         VkDescriptorSet set, uint32_t member)
{
   struct fold_set *fs = _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)set);
   if (!fs || fs->layout_handle != m->handles[member])
      return NULL;
   if (fs->instance && fs->instance->merge == m)
      return fs->instance;

   VkDescriptorSet sets[FOLD_MAX_MEMBERS] = { 0 };
   sets[member] = set;
   hash_table_u64_foreach(device->fold_set_table, entry) {
      struct fold_set *other = entry.data;
      if (other->pool != fs->pool)
         continue;
      for (uint32_t k = 0; k < m->member_count; k++)
         if (!sets[k] && k != member && other->layout_handle == m->handles[k])
            sets[k] = (VkDescriptorSet)(uintptr_t)entry.key;
   }
   for (uint32_t k = 0; k < m->member_count; k++)
      if (!sets[k]) {
         WRAPPER_LOG(error, "Folded set %u has no sibling in its pool", k);
         return NULL;
      }
   return fold_instance_create(device, m, sets);
}

bool
wrapper_fold_bind(struct wrapper_command_buffer *wcb, VkPipelineBindPoint bind_point,
                  VkPipelineLayout layout, uint32_t first, uint32_t count,
                  const VkDescriptorSet *sets, uint32_t dyn_count, const uint32_t *dyn)
{
   struct wrapper_device *device = wcb->device;
   struct fold_pipeline_layout *pl = fold_pl_lookup(device, layout);
   if (!pl)
      return false;

   VkDescriptorSet phys[32];
   uint32_t idx[32];
   uint32_t n = 0;
   uint32_t run_end = pl->run_start + pl->run_len;
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < count && n < ARRAY_SIZE(phys); i++) {
      uint32_t s = first + i;
      VkDescriptorSet set = sets[i];
      if (s >= pl->run_start && s < run_end) {
         if (n && idx[n - 1] == pl->run_start)
            continue;
         struct fold_instance *inst =
            fold_instance_for_member(device, pl->merge, set, s - pl->run_start);
         if (!inst)
            continue;
         set = inst->set;
      }
      idx[n] = fold_index(pl, s);
      phys[n++] = set;
   }
   simple_mtx_unlock(&device->fold_mutex);

   uint32_t start = 0;
   bool offsets_used = false;
   for (uint32_t i = 1; i <= n; i++) {
      if (i == n || idx[i] != idx[i - 1] + 1) {
         device->dispatch_table.CmdBindDescriptorSets(wcb->dispatch_handle, bind_point, layout,
            idx[start], i - start, &phys[start],
            offsets_used ? 0 : dyn_count, offsets_used ? NULL : dyn);
         offsets_used = true;
         start = i;
      }
   }
   return true;
}
