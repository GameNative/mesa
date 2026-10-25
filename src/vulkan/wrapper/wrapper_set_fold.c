#include <string.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "vk_util.h"
#include "util/hash_table.h"
#include "util/u_math.h"

#define FOLD_MAX_MEMBERS 8
#define FOLD_MAX_BINDINGS 16

struct fold_binding {
   VkDescriptorSetLayoutBinding vk;
   VkDescriptorBindingFlags flags;
};

struct fold_layout {
   VkDescriptorSetLayout handle;
   VkDescriptorSetLayoutCreateFlags flags;
   uint32_t binding_count;
   struct fold_binding bindings[FOLD_MAX_BINDINGS];
   uint32_t variable_binding;
   struct fold_group *group;
   uint32_t member;
   uint32_t base;
};

struct fold_group {
   uint32_t member_count;
   struct fold_layout *members[FOLD_MAX_MEMBERS];
   bool built;
   bool closed;
   VkDescriptorSetLayout merged;
   uint32_t binding_count;
   VkDescriptorSetLayoutBinding bindings[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   VkDescriptorBindingFlags binding_flags[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   uint32_t declared_variable;
};

struct fold_instance {
   struct fold_group *group;
   uint32_t used;
   uint32_t live;
   uint32_t variable_count;
   VkDescriptorPool pool;
   VkDescriptorSet set;
   struct fold_instance *next;
};

struct fold_pool {
   struct fold_instance *instances;
};

struct fold_set {
   VkDescriptorPool pool;
   struct fold_instance *instance;
   struct fold_layout *layout;
};

struct fold_pipeline_layout {
   uint32_t run_start;
   uint32_t run_len;
   struct fold_group *group;
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
   device->fold_pool_table = _mesa_hash_table_u64_create(NULL);
}

void
wrapper_fold_note_pool(struct wrapper_device *device, const VkDescriptorPoolCreateInfo *info)
{
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

   if (flags_info && info->bindingCount <= FOLD_MAX_BINDINGS)
      for (uint32_t i = 0; i < flags_info->bindingCount && i < info->bindingCount; i++)
         if (flags_info->pBindingFlags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
            variable = i;
   if (variable < 0 || !is_view_type(info->pBindings[variable].descriptorType)) {
      simple_mtx_lock(&device->fold_mutex);
      if (device->fold_open_group && device->fold_open_group->member_count)
         device->fold_open_group->closed = true;
      simple_mtx_unlock(&device->fold_mutex);
      return;
   }

   struct fold_layout *l = calloc(1, sizeof(*l));
   if (!l)
      return;
   l->handle = handle;
   l->flags = info->flags;
   l->binding_count = info->bindingCount;
   l->variable_binding = variable;
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      l->bindings[i].vk = info->pBindings[i];
      l->bindings[i].vk.pImmutableSamplers = NULL;
      l->bindings[i].flags = i < flags_info->bindingCount ? flags_info->pBindingFlags[i] : 0;
   }

   simple_mtx_lock(&device->fold_mutex);
   struct fold_group *g = device->fold_open_group;
   if (!g || g->built || g->closed || g->member_count == FOLD_MAX_MEMBERS) {
      g = calloc(1, sizeof(*g));
      device->fold_open_group = g;
   }
   if (g) {
      uint32_t base = 0;
      for (uint32_t m = 0; m < g->member_count; m++) {
         uint32_t top = 0;
         for (uint32_t b = 0; b < g->members[m]->binding_count; b++)
            top = MAX2(top, g->members[m]->bindings[b].vk.binding + 1);
         base += top;
      }
      l->group = g;
      l->member = g->member_count;
      l->base = base;
      g->members[g->member_count++] = l;
   }
   _mesa_hash_table_u64_insert(device->fold_layout_table, (uint64_t)handle, l);
   simple_mtx_unlock(&device->fold_mutex);
}

static struct fold_layout *
fold_layout_lookup(struct wrapper_device *device, VkDescriptorSetLayout handle)
{
   return _mesa_hash_table_u64_search(device->fold_layout_table, (uint64_t)handle);
}

static VkResult
fold_group_build(struct wrapper_device *device, struct fold_group *g)
{
   if (g->built)
      return g->merged ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
   g->built = true;

   uint32_t fixed = MAX2(device->fold_max_pool_count, 262144);
   VkDescriptorSetLayoutCreateFlags flags = 0;
   uint32_t n = 0;
   uint64_t total = 0;

   for (uint32_t m = 0; m < g->member_count; m++) {
      struct fold_layout *l = g->members[m];
      bool last = m == g->member_count - 1;
      flags |= l->flags;
      for (uint32_t b = 0; b < l->binding_count; b++) {
         VkDescriptorSetLayoutBinding vb = l->bindings[b].vk;
         VkDescriptorBindingFlags bf = l->bindings[b].flags;
         vb.binding += l->base;
         if (b == l->variable_binding) {
            if (last) {
               g->declared_variable = vb.descriptorCount;
            } else {
               vb.descriptorCount = MIN2(vb.descriptorCount, fixed);
               bf &= ~VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
               total += vb.descriptorCount;
            }
         } else {
            total += vb.descriptorCount;
         }
         g->bindings[n] = vb;
         g->binding_flags[n] = bf;
         n++;
      }
   }
   g->binding_count = n;

   VkDescriptorSetLayoutBindingFlagsCreateInfo bfi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = n,
      .pBindingFlags = g->binding_flags,
   };
   VkDescriptorSetLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &bfi,
      .flags = flags,
      .bindingCount = n,
      .pBindings = g->bindings,
   };
   VkResult r = device->dispatch_table.CreateDescriptorSetLayout(device->dispatch_handle,
                                                                 &ci, NULL, &g->merged);
   if (r != VK_SUCCESS)
      g->merged = VK_NULL_HANDLE;
   WRAPPER_LOG(info, "Folded %u bindless set layouts into one: %u bindings, fixed arrays of %u "
               "(largest view heap seen %u), %llu fixed descriptors, result %d",
               g->member_count, n, fixed, device->fold_max_pool_count,
               (unsigned long long)total, r);
   return r;
}

static uint32_t
fold_member_count(const struct fold_group *g, const struct fold_layout *l, uint32_t binding)
{
   for (uint32_t b = 0; b < l->binding_count; b++)
      if (l->bindings[b].vk.binding == binding) {
         for (uint32_t i = 0; i < g->binding_count; i++)
            if (g->bindings[i].binding == binding + l->base)
               return g->bindings[i].descriptorCount;
      }
   return 0;
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

   simple_mtx_lock(&device->fold_mutex);
   struct fold_group *group = NULL;
   uint32_t start = 0;
   for (uint32_t i = 0; i < info->setLayoutCount && !group; i++) {
      struct fold_layout *l = fold_layout_lookup(device, info->pSetLayouts[i]);
      if (!l || !l->group || l->member != 0 || i + l->group->member_count > info->setLayoutCount)
         continue;
      bool match = true;
      for (uint32_t m = 0; m < l->group->member_count; m++)
         match &= info->pSetLayouts[i + m] == l->group->members[m]->handle;
      if (match) {
         group = l->group;
         start = i;
      }
   }
   VkResult r = group ? fold_group_build(device, group) : VK_ERROR_INITIALIZATION_FAILED;
   simple_mtx_unlock(&device->fold_mutex);

   if (!group || r != VK_SUCCESS) {
      WRAPPER_LOG(error, "Pipeline layout needs %u sets, driver has %u, nothing to fold",
                  info->setLayoutCount, max_sets);
      return false;
   }

   uint32_t len = group->member_count;
   uint32_t count = info->setLayoutCount - len + 1;
   VkDescriptorSetLayout sets[32];
   if (count > ARRAY_SIZE(sets))
      return false;
   for (uint32_t i = 0, o = 0; i < info->setLayoutCount; i++) {
      if (i == start)
         sets[o++] = group->merged;
      else if (i < start || i >= start + len)
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
      pl->run_start = start;
      pl->run_len = len;
      pl->group = group;
      simple_mtx_lock(&device->fold_mutex);
      _mesa_hash_table_u64_insert(device->fold_pl_table, (uint64_t)*out, pl);
      simple_mtx_unlock(&device->fold_mutex);
   }
   WRAPPER_LOG(info, "Folding active for pipeline layout %u sets -> %u (sets %u-%u merged)",
               info->setLayoutCount, count, start, start + len - 1);
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

uint32_t
wrapper_fold_set_index(struct wrapper_device *device, VkPipelineLayout layout, uint32_t set)
{
   struct fold_pipeline_layout *pl = fold_pl_lookup(device, layout);
   if (!pl || set < pl->run_start)
      return set;
   if (set < pl->run_start + pl->run_len)
      return pl->run_start;
   return set - pl->run_len + 1;
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
         uint32_t id = code[i + 1];
         uint32_t s = set_of[id];
         if (s != UINT32_MAX && s >= pl->run_start) {
            bool in_run = s < pl->run_start + pl->run_len;
            if (code[i + 2] == 34)
               out[i + 3] = in_run ? pl->run_start : s - pl->run_len + 1;
            else if (code[i + 2] == 33 && in_run)
               out[i + 3] = code[i + 3] + pl->group->members[s - pl->run_start]->base;
         }
      }
      i += n;
   }
   free(set_of);
   return out;
}

static struct fold_set *
fold_set_lookup(struct wrapper_device *device, VkDescriptorSet set)
{
   return _mesa_hash_table_u64_search(device->fold_set_table, (uint64_t)set);
}

static VkResult
fold_instance_realize(struct wrapper_device *device, struct fold_instance *inst)
{
   if (inst->set)
      return VK_SUCCESS;
   struct fold_group *g = inst->group;
   VkResult r = fold_group_build(device, g);
   if (r != VK_SUCCESS)
      return r;

   VkDescriptorPoolSize sizes[FOLD_MAX_MEMBERS * FOLD_MAX_BINDINGS];
   uint32_t size_count = 0;
   uint32_t variable = MIN2(inst->variable_count, g->declared_variable);
   uint64_t total = 0;
   for (uint32_t i = 0; i < g->binding_count; i++) {
      uint32_t c = (g->binding_flags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         ? variable : g->bindings[i].descriptorCount;
      uint32_t j;
      for (j = 0; j < size_count; j++)
         if (sizes[j].type == g->bindings[i].descriptorType)
            break;
      if (j == size_count) {
         sizes[size_count].type = g->bindings[i].descriptorType;
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
   r = device->dispatch_table.CreateDescriptorPool(device->dispatch_handle, &pci, NULL,
                                                   &inst->pool);
   if (r != VK_SUCCESS)
      return r;
   VkDescriptorSetVariableDescriptorCountAllocateInfo vai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
      .descriptorSetCount = 1,
      .pDescriptorCounts = &variable,
   };
   VkDescriptorSetAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = &vai,
      .descriptorPool = inst->pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &g->merged,
   };
   r = device->dispatch_table.AllocateDescriptorSets(device->dispatch_handle, &ai, &inst->set);
   if (r != VK_SUCCESS) {
      device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, inst->pool, NULL);
      inst->pool = VK_NULL_HANDLE;
      inst->set = VK_NULL_HANDLE;
   }
   WRAPPER_LOG(info, "Folded descriptor set allocated: %llu descriptors (variable %u), result %d",
               (unsigned long long)total, variable, r);
   return r;
}

static void
fold_instance_free(struct wrapper_device *device, struct fold_instance *inst)
{
   if (inst->pool)
      device->dispatch_table.DestroyDescriptorPool(device->dispatch_handle, inst->pool, NULL);
   free(inst);
}

bool
wrapper_fold_allocate(struct wrapper_device *device, const VkDescriptorSetAllocateInfo *info,
                      VkDescriptorSet *sets, VkResult *result)
{
   bool any = false;
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
      struct fold_layout *l = fold_layout_lookup(device, info->pSetLayouts[i]);
      any |= l && l->group;
   }
   simple_mtx_unlock(&device->fold_mutex);
   if (!any)
      return false;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *vai =
      vk_find_struct_const(info->pNext, DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);
   *result = VK_SUCCESS;
   for (uint32_t i = 0; i < info->descriptorSetCount; i++)
      sets[i] = VK_NULL_HANDLE;

   for (uint32_t i = 0; i < info->descriptorSetCount && *result == VK_SUCCESS; i++) {
      uint32_t variable = vai && i < vai->descriptorSetCount ? vai->pDescriptorCounts[i] : 0;
      simple_mtx_lock(&device->fold_mutex);
      struct fold_layout *l = fold_layout_lookup(device, info->pSetLayouts[i]);
      simple_mtx_unlock(&device->fold_mutex);

      if (!l || !l->group) {
         VkDescriptorSetVariableDescriptorCountAllocateInfo one_vai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
            .descriptorSetCount = 1,
            .pDescriptorCounts = &variable,
         };
         VkDescriptorSetAllocateInfo one = *info;
         one.pNext = vai ? &one_vai : NULL;
         one.descriptorSetCount = 1;
         one.pSetLayouts = &info->pSetLayouts[i];
         *result = device->dispatch_table.AllocateDescriptorSets(device->dispatch_handle,
                                                                 &one, &sets[i]);
         continue;
      }

      struct fold_set *fs = calloc(1, sizeof(*fs));
      if (!fs) {
         *result = VK_ERROR_OUT_OF_HOST_MEMORY;
         break;
      }
      simple_mtx_lock(&device->fold_mutex);
      struct fold_pool *fp =
         _mesa_hash_table_u64_search(device->fold_pool_table, (uint64_t)info->descriptorPool);
      if (!fp && (fp = calloc(1, sizeof(*fp))))
         _mesa_hash_table_u64_insert(device->fold_pool_table, (uint64_t)info->descriptorPool, fp);
      struct fold_instance *inst = NULL;
      if (fp) {
         for (inst = fp->instances; inst; inst = inst->next)
            if (inst->group == l->group && !(inst->used & BITFIELD_BIT(l->member)))
               break;
         if (!inst && (inst = calloc(1, sizeof(*inst)))) {
            inst->group = l->group;
            inst->next = fp->instances;
            fp->instances = inst;
         }
      }
      if (inst) {
         inst->used |= BITFIELD_BIT(l->member);
         inst->live++;
         inst->variable_count = MAX2(inst->variable_count, variable);
         fs->pool = info->descriptorPool;
         fs->instance = inst;
         fs->layout = l;
         sets[i] = (VkDescriptorSet)(uintptr_t)fs;
         _mesa_hash_table_u64_insert(device->fold_set_table, (uint64_t)sets[i], fs);
      } else {
         free(fs);
         *result = VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      simple_mtx_unlock(&device->fold_mutex);
   }
   return true;
}

static void
fold_release_set(struct wrapper_device *device, struct fold_set *fs)
{
   struct fold_instance *inst = fs->instance;
   inst->used &= ~BITFIELD_BIT(fs->layout->member);
   if (--inst->live == 0) {
      struct fold_pool *fp =
         _mesa_hash_table_u64_search(device->fold_pool_table, (uint64_t)fs->pool);
      if (fp) {
         struct fold_instance **p = &fp->instances;
         while (*p && *p != inst)
            p = &(*p)->next;
         if (*p)
            *p = inst->next;
      }
      fold_instance_free(device, inst);
   }
   free(fs);
}

uint32_t
wrapper_fold_free_sets(struct wrapper_device *device, uint32_t count, const VkDescriptorSet *sets,
                       VkDescriptorSet *real)
{
   uint32_t n = 0;
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < count; i++) {
      struct fold_set *fs = sets[i] ? fold_set_lookup(device, sets[i]) : NULL;
      if (fs) {
         _mesa_hash_table_u64_remove(device->fold_set_table, (uint64_t)sets[i]);
         fold_release_set(device, fs);
      } else if (sets[i]) {
         real[n++] = sets[i];
      }
   }
   simple_mtx_unlock(&device->fold_mutex);
   return n;
}

void
wrapper_fold_reset_pool(struct wrapper_device *device, VkDescriptorPool pool, bool destroy)
{
   simple_mtx_lock(&device->fold_mutex);
   hash_table_u64_foreach(device->fold_set_table, entry) {
      struct fold_set *fs = entry.data;
      if (fs->pool == pool) {
         _mesa_hash_table_u64_remove(device->fold_set_table, entry.key);
         fold_release_set(device, fs);
      }
   }
   if (destroy) {
      struct fold_pool *fp = _mesa_hash_table_u64_search(device->fold_pool_table, (uint64_t)pool);
      if (fp) {
         _mesa_hash_table_u64_remove(device->fold_pool_table, (uint64_t)pool);
         free(fp);
      }
   }
   simple_mtx_unlock(&device->fold_mutex);
}

static bool
fold_resolve(struct wrapper_device *device, VkDescriptorSet *set, uint32_t *binding,
             uint32_t *element, uint32_t *count)
{
   struct fold_set *fs = fold_set_lookup(device, *set);
   if (!fs)
      return true;
   if (fold_instance_realize(device, fs->instance) != VK_SUCCESS)
      return false;
   uint32_t limit = fold_member_count(fs->instance->group, fs->layout, *binding);
   if (fs->layout->bindings[fs->layout->variable_binding].vk.binding == *binding &&
       fs->layout->member == fs->instance->group->member_count - 1)
      limit = MIN2(fs->instance->variable_count, fs->instance->group->declared_variable);
   if (*element >= limit) {
      static bool warned;
      if (!warned) {
         warned = true;
         WRAPPER_LOG(error, "Folded descriptor write past the fixed array size %u dropped", limit);
      }
      return false;
   }
   *count = MIN2(*count, limit - *element);
   *binding += fs->layout->base;
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
   for (uint32_t i = 0; i < *write_count && !touched; i++)
      touched = fold_set_lookup(device, (*writes)[i].dstSet) != NULL;
   for (uint32_t i = 0; i < *copy_count && !touched; i++)
      touched = fold_set_lookup(device, (*copies)[i].dstSet) ||
                fold_set_lookup(device, (*copies)[i].srcSet);
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
   simple_mtx_lock(&device->fold_mutex);
   for (uint32_t i = 0; i < count && n < ARRAY_SIZE(phys); i++) {
      uint32_t s = first + i;
      VkDescriptorSet set = sets[i];
      struct fold_set *fs = fold_set_lookup(device, set);
      if (s >= pl->run_start && s < pl->run_start + pl->run_len) {
         if (n && idx[n - 1] == pl->run_start)
            continue;
         if (!fs || fold_instance_realize(device, fs->instance) != VK_SUCCESS)
            continue;
         set = fs->instance->set;
      }
      idx[n] = s < pl->run_start ? s
             : s < pl->run_start + pl->run_len ? pl->run_start : s - pl->run_len + 1;
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
