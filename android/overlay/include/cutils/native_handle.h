#pragma once
/*
 * Minimal cutils/native_handle.h stub for the Android cross-build of this
 * Mesa tree.  include/vulkan/vk_android_native_buffer.h includes it whenever
 * __ANDROID__ is defined (any NDK clang build), but cutils is the Android
 * platform libcutils and is absent from the NDK sysroot.
 *
 * Only the native_handle_t type is needed by the sources we build; the
 * native_handle_create()/close() helpers live in libcutils and are not used
 * on this (no Android platform) path.  vk_android_native_buffer.h defines
 * buffer_handle_t itself when ANDROID_API_LEVEL < 28, so we deliberately do
 * not define it here.
 *
 * android/overlay/include/cutils/trace.h for the rationale.
 */
#ifndef GN_CUTILS_NATIVE_HANDLE_STUB_H
#define GN_CUTILS_NATIVE_HANDLE_STUB_H

typedef struct native_handle {
   int version;
   int numFds;
   int numInts;
   int data[0];
} native_handle_t;

#endif /* GN_CUTILS_NATIVE_HANDLE_STUB_H */
