#pragma once
/*
 * Minimal cutils/trace.h stub for the Android cross-build of this Mesa tree.
 *
 * Mesa's src/util/perf/cpu_trace.h includes <cutils/trace.h> and calls
 * atrace_begin()/atrace_end() whenever DETECT_OS_ANDROID is set — i.e. for
 * every NDK clang build, because the compiler defines __ANDROID__ regardless
 * of the cross file's host_machine.system.  cutils is an Android *platform*
 * library and its headers are not part of the NDK sysroot (nor of the termux
 * prefix), so the include fails unless something provides it.
 *
 * Tracing here is inert anyway: perfetto is disabled in this build and
 * atrace is only ever called when perfetto/tracing is active.  This stub
 * only needs to satisfy the compiler.  It is added to the include path from
 * android/cross/aarch64-linux-android.txt.
 */
#ifndef GN_CUTILS_TRACE_STUB_H
#define GN_CUTILS_TRACE_STUB_H

#define ATRACE_TAG_GRAPHICS (1u << 1)

#define atrace_begin(tag, name)                  ((void)(tag), (void)(name))
#define atrace_end(tag)                          ((void)(tag))
#define atrace_async_begin(tag, name, cookie)    ((void)(tag), (void)(name), (void)(cookie))
#define atrace_async_end(tag, name, cookie)      ((void)(tag), (void)(name), (void)(cookie))
#define atrace_init()                            ((void)0)

#endif /* GN_CUTILS_TRACE_STUB_H */
