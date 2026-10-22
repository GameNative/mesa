#!/bin/bash
# Usage: NDK=<ndk 27.3> TERMUX_ROOT=<extracted termux aarch64 rootfs> ./configure.sh <build dir> [meson args]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
BUILD=${1:?build dir}; shift
: "${NDK:?}" "${TERMUX_ROOT:?}"
HOST_TAG=${HOST_TAG:-darwin-x86_64}
TC=$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin
T=$TERMUX_ROOT/data/data/com.termux/files/usr
mkdir -p "$BUILD"
sed -e "s|@TC@|$TC|g" -e "s|@T@|$T|g" -e "s|@PKGCONFIG@|$HERE/pkg-config|g" \
  "$HERE/cross-android.txt.in" > "$BUILD/cross-android.txt"
unset PKG_CONFIG_PATH PKG_CONFIG_LIBDIR
export TERMUX_ROOT
cd "$SRC"
meson setup "$BUILD" --cross-file "$BUILD/cross-android.txt" --prefix=/usr --libdir=lib \
  -Dgallium-drivers=zink -Dvulkan-drivers= -Dglx=dri -Degl=enabled -Dgbm=disabled -Dplatforms=x11 \
  -Degl-native-platform=x11 -Dgles1=disabled -Dgles2=enabled -Dshared-glapi=enabled -Dglvnd=disabled \
  -Dllvm=disabled -Ddri3=enabled -Dopengl=true -Dbuildtype=release -Db_ndebug=true -Dzstd=enabled \
  -Dshader-cache=enabled -Dvideo-codecs= -Dtools= -Dxmlconfig=disabled -Dosmesa=false \
  -Dandroid-libbacktrace=disabled -Dandroid-strict=false -Dlibunwind=disabled -Dvalgrind=disabled \
  -Dlmsensors=disabled -Dbuild-tests=false -Dgallium-va=disabled -Dgallium-vdpau=disabled \
  -Dgallium-xa=disabled -Dgallium-nine=false -Dgallium-opencl=disabled -Dgallium-rusticl=false \
  -Dselinux=false -Dperfetto=false -Dexpat=disabled -Dplatform-sdk-version=28 "$@"
