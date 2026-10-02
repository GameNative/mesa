#pragma once
/*
 * Minimal log/log.h stub for the Android cross-build of this Mesa tree.
 *
 * src/util/os_misc.c includes <log/log.h> whenever DETECT_OS_ANDROID is set
 * (every NDK clang build defines __ANDROID__).  log/ is the Android *platform*
 * liblog, not in the NDK sysroot.  Only LOG_PRI + ANDROID_LOG_* are needed;
 * the NDK's own <android/log.h> supplies the rest.  See
 * android/overlay/include/cutils/trace.h for the rationale.
 */
#ifndef GN_LOG_LOG_STUB_H
#define GN_LOG_LOG_STUB_H

#include <android/log.h>

/* From system/core/liblog, prefixing the liblog tag when logging verbosely. */
#ifndef LOG_PRI
#define LOG_PRI(priority, tag, ...) __android_log_print(priority, tag, __VA_ARGS__)
#endif

#endif /* GN_LOG_LOG_STUB_H */
