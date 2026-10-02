#!/usr/bin/env bash
# build-mesa-glamor.sh — NDK cross-build of a minimal Unix GL/EGL stack for
# Xwayland's glamor, out of this Mesa tree.
#
# Produces (into a dedicated staging prefix, NOT the shared termux prefix):
#   libEGL_mesa.so.0        EGL vendor library (GBM platform: EGL_PLATFORM_GBM_KHR)
#   libGLESv2.so.2          GLES2 entrypoints
#   libgallium-<ver>.so     gallium (zink + softpipe only)
#   libgbm.so               GBM
#   lib/gbm/dri_gbm.so      GBM "dri" backend (dlopen'd by libgbm.so)
#   dri/zink_dri.so         zink  (Vulkan-backed gallium driver)
#   dri/softpipe_dri.so     softpipe (software fallback)
#   share/glvnd/egl_vendor.d/50_mesa.json   glvnd EGL vendor descriptor
#
# This is deliberately NOT built into $HOME/termuxfs/.../usr: the 54 MB,
# LLVM-bound Mesa 25.3.5 already installed there must stay intact as the
# working fallback.  Install prefix defaults to
#   <mesa>/build-android-glamor/stage
# and is overridable with GN_MESA_GLAMOR_STAGE.
#
# Model: the gndrm socket-backed render-node fd support is committed in this
# tree, so there is nothing to patch — this script configures and builds the
# in-tree sources directly.  Local Android support is just this script plus
# the header/pkg-config overlay in android/overlay/.
#
# Env overrides:
#   GN_MESA_NDK            Android NDK root (default: $HOME/Android/Sdk/ndk/27.3.13750724)
#   GN_MESA_DEPS_PREFIX    build-time aarch64 sysroot (default: termux prefix)
#   GN_MESA_GLAMOR_STAGE   install prefix for the artifacts (default below)
#   GN_MESA_GLAMOR_BUILD   meson build dir (default: <mesa>/build-android-glamor)
#   GNDRM_DRM_NAME         device name gndrm reports (baked default: gamenative)
#   GNDRM_MESA_DRIVER      DRI driver chosen for the gndrm fd (default: zink)

set -euo pipefail

MESA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="${GN_MESA_DEPS_PREFIX:-$HOME/termuxfs/aarch64/data/data/com.termux/files/usr}"
BUILD_DIR="${GN_MESA_GLAMOR_BUILD:-$MESA_DIR/build-android-glamor}"
STAGE="${GN_MESA_GLAMOR_STAGE:-$BUILD_DIR/stage}"
NDK="${GN_MESA_NDK:-$HOME/Android/Sdk/ndk/27.3.13750724}"

# --- cross file --------------------------------------------------------------
# android/cross/aarch64-linux-android.txt is generated from the committed .txt.in
# template and the settings above, so no absolute path is stored in the tree.
# Regenerated on every run; meson reads it at every setup/reconfigure.
CROSS_IN="$MESA_DIR/android/cross/aarch64-linux-android.txt.in"
CROSS_OUT="$MESA_DIR/android/cross/aarch64-linux-android.txt"
sed -e "s|@NDK_BIN@|$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin|g" \
    -e "s|@DEPS@|$DEPS|g" \
    -e "s|@REPO_DIR@|$MESA_DIR|g" \
    "$CROSS_IN" > "$CROSS_OUT"
if grep -q '@NDK_BIN@\|@DEPS@\|@REPO_DIR@' "$CROSS_OUT"; then
    echo "cross file still has unsubstituted placeholders: $CROSS_OUT" >&2
    exit 1
fi

# --- source -----------------------------------------------------------------
# The gndrm socket-backed render-node fd support (loader/dri3) is committed in
# this tree, so no patches are applied: the build reads the in-tree sources
# directly.

# --- meson ------------------------------------------------------------------
# Minimal stack for Xwayland glamor/dri3 against gndrm:
#   platforms=            no X11/Wayland platform (glamor talks EGL+GBM only)
#   gbm=enabled           -> EGL gains the GBM platform (platform_drm.c)
#   gallium-drivers=zink,softpipe    only these two (no llvmpipe/swrast/kms_swrast)
#   vulkan-drivers=       none (turnip ships separately as libvulkan_freedreno.so)
#   llvm=disabled         keeps libLLVM.so (128 MB) out of libgallium
#   glvnd=enabled         EGL built as libEGL_mesa.so.0 + 50_mesa.json
#   xmlconfig/expat off   drop the libexpat dependency (driconf hardcoded)
#   zstd off              drop libzstd; no disk-cache compression
cd "$MESA_DIR"
RECONF=""
if [ -d "$BUILD_DIR/meson-info" ]; then
    RECONF="--reconfigure"
elif [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
meson setup "$BUILD_DIR" \
    --cross-file "$MESA_DIR/android/cross/aarch64-linux-android.txt" \
    -Dprefix="$STAGE" \
    -Dbuildtype=release \
    -Dplatforms= \
    -Degl=enabled -Dgbm=enabled \
    -Dgles2=enabled -Dgles1=disabled -Dopengl=false -Dglx=disabled \
    -Dgallium-drivers=zink,softpipe \
    -Dvulkan-drivers= \
    -Dshared-glapi=enabled \
    -Dglvnd=enabled \
    -Dllvm=disabled \
    -Dvalgrind=disabled \
    -Dzstd=disabled \
    -Dxmlconfig=disabled -Dexpat=disabled \
    -Dgallium-va=disabled -Dgallium-vdpau=disabled -Dgallium-xa=disabled \
    -Dgallium-opencl=disabled -Dgallium-rusticl=false \
    -Dgallium-d3d10umd=false -Dgallium-nine=false \
    -Dosmesa=false -Dtools= \
    -Dbuild-tests=false -Dlmsensors=disabled -Dperfetto=false \
    -Dandroid-libbacktrace=disabled \
    -Dinstall-intel-clc=false \
    $RECONF

ninja -C "$BUILD_DIR"

# --- install to the dedicated prefix ---------------------------------------
rm -rf "$STAGE"
ninja -C "$BUILD_DIR" install

echo
echo "=== installed (staged prefix: $STAGE) ==="
find "$STAGE" -type f -o -type l | sort
echo
echo "=== sizes ==="
du -sh "$STAGE"
find "$STAGE" -type f -name '*.so*' -exec du -b {} + | sort -n
echo
echo "Note for the packager: GBM backend and gallium pipe directories are"
echo "resolved at runtime as siblings of the library that loads them"
echo "(libgbm.so -> lib/gbm/dri_gbm.so), so no GBM_BACKENDS_PATH /"
echo "LIBGL_DRIVERS_PATH absolute-path wiring is required. The glvnd loader"
echo "libEGL.so.1 + libGLdispatch.so.0 come from libglvnd (already in the"
echo "shared termux prefix), not from this build."
