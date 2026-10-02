#pragma once
/*
 * Minimal cutils/properties.h stub for the Android cross-build of this Mesa
 * tree.  src/util/os_misc.c includes it under DETECT_OS_ANDROID (any NDK
 * clang build) and calls property_get() with PROPERTY_VALUE_MAX /
 * PROPERTY_KEY_MAX.  cutils is the Android platform libcutils, absent from
 * the NDK sysroot; bionic libc already provides __system_property_get(), so
 * property_get() is a header-only wrapper over it.
 *
 * See android/overlay/include/cutils/trace.h for the rationale.
 */
#ifndef GN_CUTILS_PROPERTIES_STUB_H
#define GN_CUTILS_PROPERTIES_STUB_H

#include <string.h>
#include <sys/system_properties.h>

#ifndef PROPERTY_VALUE_MAX
#define PROPERTY_VALUE_MAX PROP_VALUE_MAX
#endif
#ifndef PROPERTY_KEY_MAX
#define PROPERTY_KEY_MAX PROP_NAME_MAX
#endif

static inline int
property_get(const char *key, char *value, const char *default_value)
{
   int len = __system_property_get(key, value);
   if (len <= 0 && default_value) {
      strlcpy(value, default_value, PROPERTY_VALUE_MAX);
      len = (int)strlen(value);
   }
   return len;
}

#endif /* GN_CUTILS_PROPERTIES_STUB_H */
