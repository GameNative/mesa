#ifndef __SPIRV_PATCHER_HPP
#define __SPIRV_PATCHER_HPP

#ifdef __cplusplus
extern "C" {
#endif

void
remove_ClipDistance_CullDistance(uint32_t *pCode, size_t *codeSize);

void
patch_OpConstantComposite_to_OpSpecConstantComposite(uint32_t *pCode, uint32_t codeSize);

#ifdef __cplusplus
}
#endif

#endif
