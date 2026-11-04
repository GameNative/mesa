#!/bin/bash
# Usage: ./build.sh <build dir> [ninja args]
BUILD=${1:?build dir}; shift
unset PKG_CONFIG_PATH PKG_CONFIG_LIBDIR
export TERMUX_ROOT
ninja -C "$BUILD" -k 0 "$@"
