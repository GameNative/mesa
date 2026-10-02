# android/ — aarch64-linux-android builds of this Mesa tree

Two independent Android builds live here; neither disturbs the other.

## `build-mesa-glamor.sh` — minimal Unix GL/EGL stack for Xwayland glamor

Run as `cd <mesa> && bash android/build-mesa-glamor.sh`.

Produces a **dedicated staging prefix** (default
`<mesa>/build-android-glamor/stage`, override `GN_MESA_GLAMOR_STAGE`), leaving
the shared termux prefix (`$HOME/termuxfs/aarch64/.../usr`, Mesa 25.3.5,
54 MB, LLVM-bound) intact as a fallback.

Enabled: EGL (platforms `surfaceless` + `drm` i.e. `EGL_PLATFORM_GBM_KHR`),
GBM, GLES2, shared-glapi, glvnd, gallium with **zink + softpipe**.
Disabled: LLVM (no `libLLVM.so`/`libicudata` in the closure), Vulkan drivers
(turnip ships separately), GLX/X11, OSmesa, VA/VDPAU/XA, zstd, expat/driconf.

Layout (mirrors the shipped termux-prefix Mesa):

| path | what |
|---|---|
| `lib/libEGL_mesa.so.0.0.0` | EGL vendor library (glvnd) |
| `lib/libgallium-25.0.0-devel.so` | gallium megadriver (zink + softpipe) |
| `lib/libgbm.so.1.0.0` | GBM |
| `lib/libglapi.so.0.0.0` | shared glapi |
| `lib/gbm/dri_gbm.so` | GBM "dri" backend |
| `lib/dri/libdril_dri.so` (+ `zink_dri.so`, `swrast_dri.so`, `kms_swrast_dri.so` symlinks) | DRI/GLX compat megadriver |
| `share/glvnd/egl_vendor.d/50_mesa.json` | glvnd EGL vendor descriptor |

There is **no** `dri/softpipe_dri.so` — Mesa exposes softpipe under the DRI
driver name `swrast` (`dri/swrast_dri.so`). There is also **no** libGLESv2 /
libEGL.so.1 from this build: with `-Dglvnd=enabled` those loaders come from
libglvnd (present in the shared termux prefix), exactly as for the currently
shipped Mesa.

Runtime notes for the packager:
* `libgbm.so` resolves the GBM backend directory at runtime, as its own
  sibling `gbm/` directory, so it works wherever `<prefix>/lib` is unpacked —
  no `GBM_BACKENDS_PATH` needed. Likewise the gallium pipe directory is
  resolved as the sibling `gallium-pipe/` (that directory is not shipped, so
  the software pipe fallback stays absent, as before).
* glvnd needs the vendor dir: point the loader at
  `<prefix>/share/glvnd/egl_vendor.d` (e.g. `__EGL_VENDOR_LIBRARY_DIRS`).
* Android-provided dependencies: `libc`, `libdl`, `libm`, `libz`, `liblog`,
  `libsync`. `libdrm.so` comes from the termux prefix set.

Local Android changes are committed in the tree: the gndrm socket-backed
render-node fd support (loader/dri3) plus the header/pkg-config overlay in
`android/overlay/` (used only by this build). No patches are applied — the
build reads the in-tree sources directly. `*.patch` and `shims/` are in
`.gitignore`.

## `android.toml` + `../.github/workflows/build.yml`

The pre-existing Vulkan-wrapper (`libvulkan_wrapper.so`) build. Untouched.
