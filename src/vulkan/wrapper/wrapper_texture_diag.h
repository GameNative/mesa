/* Temporary, bounded Zink diagnostic; never changes rendering state.
 * Metadata is device-owned and protected by resource_mutex. Unlogged state
 * after a limit is unknown, not evidence that an operation was unused. */
struct wrapper_texture_template {
   struct wrapper_texture_template *next;
   VkDescriptorUpdateTemplate handle;
   uint32_t count;
   VkDescriptorUpdateTemplateEntry entries[];
};

static bool
texture_diag_enabled(struct wrapper_device *d)
{
   const char *e = d->physical->instance->vk.app_info.engine_name;
   return e && !strcmp(e, "mesa zink");
}

static bool
texture_diag_budget(struct wrapper_device *d, unsigned *counter,
                    unsigned limit, const char *kind)
{
   if (!texture_diag_enabled(d)) return false;
   unsigned n = p_atomic_inc_return(counter);
   if (n == limit + 1)
      fprintf(stderr, "[WRAPPER_SAMPLE] %s initial limit=%u; then every 4096th call up to call 262144; gaps are unknown\n", kind, limit);
   return n <= limit || (n <= 262144 && n % 4096 == 0);
}

static bool
texture_diag_image_type(VkDescriptorType t)
{
   return t == VK_DESCRIPTOR_TYPE_SAMPLER ||
          t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
          t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
          t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
          t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}

static void
texture_diag_image(struct wrapper_device *d, const char *path, uint64_t owner,
                   uint32_t binding, uint32_t element, VkDescriptorType t,
                   const VkDescriptorImageInfo *i)
{
   static unsigned count;
   if (!texture_diag_budget(d, &count, 2048, "descriptor-image")) return;
   /* Ignore fields not used by this descriptor type. */
   bool s = t == VK_DESCRIPTOR_TYPE_SAMPLER || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   bool v = t != VK_DESCRIPTOR_TYPE_SAMPLER;
   fprintf(stderr, "[WRAPPER_SAMPLE] %s dev=%p owner=%llx binding=%u element=%u type=%d sampler=%llx view=%llx layout=%d\n",
           path, (void *)d, (unsigned long long)owner, binding, element, t,
           s ? (unsigned long long)i->sampler : 0,
           v ? (unsigned long long)i->imageView : 0, v ? i->imageLayout : 0);
}

static void
texture_diag_writes(struct wrapper_device *d, const char *path,
                    uint32_t count, const VkWriteDescriptorSet *writes)
{
   static unsigned batches;
   if (!texture_diag_budget(d, &batches, 512, "write-batch")) return;
   for (uint32_t k = 0; k < MIN2(count, 64); k++) {
      const VkWriteDescriptorSet *w = &writes[k];
      if (!texture_diag_image_type(w->descriptorType) || !w->pImageInfo) continue;
      for (uint32_t j = 0; j < MIN2(w->descriptorCount, 32); j++)
         texture_diag_image(d, path, !strcmp(path, "push-write") ? 0 : (uint64_t)w->dstSet, w->dstBinding,
                            w->dstArrayElement + j, w->descriptorType, &w->pImageInfo[j]);
      if (w->descriptorCount > 32)
         fprintf(stderr, "[WRAPPER_SAMPLE] array truncated to 32 of %u\n", w->descriptorCount);
   }
   if (count > 64) fprintf(stderr, "[WRAPPER_SAMPLE] writes truncated to 64 of %u\n", count);
}

static void
texture_diag_template_create(struct wrapper_device *d, VkDescriptorUpdateTemplate h,
                             const VkDescriptorUpdateTemplateCreateInfo *i)
{
   static unsigned count;
   if (!texture_diag_budget(d, &count, 128, "template-create")) return;
   uint32_t n = MIN2(i->descriptorUpdateEntryCount, 64);
   struct wrapper_texture_template *r = calloc(1, sizeof(*r) + n * sizeof(r->entries[0]));
   if (!r) {
      fprintf(stderr, "[WRAPPER_SAMPLE] metadata allocation failed; template unknown\n");
      return;
   }
   r->handle = h; r->count = n;
   if (n) memcpy(r->entries, i->pDescriptorUpdateEntries, n * sizeof(r->entries[0]));
   simple_mtx_lock(&d->resource_mutex);
   r->next = d->texture_diag_templates; d->texture_diag_templates = r;
   simple_mtx_unlock(&d->resource_mutex);
   fprintf(stderr, "[WRAPPER_SAMPLE] template dev=%p handle=%llx type=%d entries=%u captured=%u\n",
           (void *)d, (unsigned long long)h, i->templateType, i->descriptorUpdateEntryCount, n);
}

static void
texture_diag_template_destroy(struct wrapper_device *d, VkDescriptorUpdateTemplate h)
{
   simple_mtx_lock(&d->resource_mutex);
   struct wrapper_texture_template **p = &d->texture_diag_templates;
   while (*p) {
      if ((*p)->handle == h) {
         struct wrapper_texture_template *r = *p; *p = r->next; free(r); break;
      }
      p = &(*p)->next;
   }
   simple_mtx_unlock(&d->resource_mutex);
}

static void
texture_diag_template_update(struct wrapper_device *d, VkDescriptorUpdateTemplate h,
                             uint64_t owner, const void *data)
{
   static unsigned count;
   if (!texture_diag_budget(d, &count, 512, "template-update")) return;
   simple_mtx_lock(&d->resource_mutex);
   struct wrapper_texture_template *r = d->texture_diag_templates;
   while (r && r->handle != h) r = r->next;
   fprintf(stderr, "[WRAPPER_SAMPLE] template-update dev=%p template=%llx owner=%llx tracked=%d\n",
           (void *)d, (unsigned long long)h, (unsigned long long)owner, r != NULL);
   if (r) for (uint32_t k = 0; k < r->count; k++) {
      const VkDescriptorUpdateTemplateEntry *e = &r->entries[k];
      if (!texture_diag_image_type(e->descriptorType)) continue;
      for (uint32_t j = 0; j < MIN2(e->descriptorCount, 32); j++) {
         VkDescriptorImageInfo image = {0};
         const char *src = (const char *)data + e->offset + j * e->stride;
         /* The caller need only provide the members used by the type. */
         if (e->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
             e->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            memcpy(&image.sampler, src + offsetof(VkDescriptorImageInfo, sampler), sizeof(image.sampler));
         if (e->descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) {
            memcpy(&image.imageView, src + offsetof(VkDescriptorImageInfo, imageView), sizeof(image.imageView));
            memcpy(&image.imageLayout, src + offsetof(VkDescriptorImageInfo, imageLayout), sizeof(image.imageLayout));
         }
         texture_diag_image(d, "template-image", owner, e->dstBinding,
                            e->dstArrayElement + j, e->descriptorType, &image);
      }
      if (e->descriptorCount > 32)
         fprintf(stderr, "[WRAPPER_SAMPLE] template array truncated to 32 of %u\n", e->descriptorCount);
   }
   simple_mtx_unlock(&d->resource_mutex);
}
