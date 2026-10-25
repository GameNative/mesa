#ifndef WRAPPER_SPIRV_LOWER_H
#define WRAPPER_SPIRV_LOWER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum wrapper_spirv_lower_result {
   WRAPPER_SPIRV_UNCHANGED,
   WRAPPER_SPIRV_LOWERED,
   WRAPPER_SPIRV_FAILED,
};

#define WRAPPER_SPIRV_STRIP_WORKGROUP_INIT (1u << 0)
#define WRAPPER_SPIRV_LOWER_DEMOTE (1u << 1)

enum wrapper_spirv_lower_result
wrapper_spirv_lower_to_1_3(const uint32_t *in, size_t words, uint32_t flags,
                           uint32_t **out_code, size_t *out_words);

#ifdef __cplusplus
}
#endif

#endif
