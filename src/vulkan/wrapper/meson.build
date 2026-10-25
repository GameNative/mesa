wrapper_icd = custom_target(
  'wrapper_icd',
  input: [vk_icd_gen, vk_api_xml],
  output: 'wrapper_icd.@0@.json'.format(host_machine.cpu()),
  command: [
    prog_python,
    '@INPUT0@',
    '--api-version', '1.3',
    '--xml', '@INPUT1@',
    '--lib-path', join_paths(get_option('prefix'), get_option('libdir'), 'libvulkan_wrapper.so'),
    '--out', '@OUTPUT@',
  ],
  build_by_default: true,
  install_dir: with_vulkan_icd_dir,
  install: true,
)

wrapper_entrypoints = custom_target(
  'wrapper_entrypoints',
  input: [vk_entrypoints_gen, vk_api_xml],
  output: ['wrapper_entrypoints.h', 'wrapper_entrypoints.c'],
  command: [
    prog_python,
    '@INPUT0@',
    '--xml', '@INPUT1@',
    '--proto',
    '--weak',
    '--beta', with_vulkan_beta.to_string(),
    '--out-h', '@OUTPUT0@',
    '--out-c', '@OUTPUT1@',
    '--prefix', 'wrapper',
  ],
  depend_files: vk_entrypoints_gen_depend_files,
)

wrapper_trampolines_gen = files('vk_wrapper_trampolines_gen.py')

wrapper_trampolines = custom_target(
  'wrapper_trampolines',
  input: [wrapper_trampolines_gen, vk_api_xml],
  output: ['wrapper_trampolines.c', 'wrapper_trampolines.h'],
  command: [
    prog_python,
    '@INPUT0@',
    '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@',
    '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files: vk_dispatch_trampolines_gen_depend_files,
)

wrapper_features_gen = files('vk_wrapper_features_gen.py')

wrapper_features = custom_target(
  'vk_wrapper_features',
  input: [wrapper_features_gen, vk_api_xml],
  output: ['wrapper_features.c'],
  command: [
    prog_python,
    '@INPUT0@',
    '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files: vk_physical_device_features_gen_depend_files,
)

adrenotools_subproj = subproject('libadrenotools')

adrenotools_dep = declare_dependency(
  include_directories: adrenotools_subproj.get_variable('adrenotools_inc'),
  link_with: adrenotools_subproj.get_variable('adrenotools_lib'),
)

wrapper_astcenc_simd = host_machine.cpu_family() == 'aarch64' ? '-DASTCENC_NEON=1' : '-DASTCENC_NEON=0'

libwrapper_astcenc = static_library(
  'wrapper_astcenc',
  files(
  'astcenc/astcenc_averages_and_directions.cpp',
  'astcenc/astcenc_block_sizes.cpp',
  'astcenc/astcenc_color_quantize.cpp',
  'astcenc/astcenc_color_unquantize.cpp',
  'astcenc/astcenc_compress_symbolic.cpp',
  'astcenc/astcenc_compute_variance.cpp',
  'astcenc/astcenc_decompress_symbolic.cpp',
  'astcenc/astcenc_diagnostic_trace.cpp',
  'astcenc/astcenc_entry.cpp',
  'astcenc/astcenc_find_best_partitioning.cpp',
  'astcenc/astcenc_ideal_endpoints_and_weights.cpp',
  'astcenc/astcenc_image.cpp',
  'astcenc/astcenc_integer_sequence.cpp',
  'astcenc/astcenc_mathlib.cpp',
  'astcenc/astcenc_mathlib_softfloat.cpp',
  'astcenc/astcenc_partition_tables.cpp',
  'astcenc/astcenc_percentile_tables.cpp',
  'astcenc/astcenc_pick_best_endpoint_format.cpp',
  'astcenc/astcenc_quantization.cpp',
  'astcenc/astcenc_symbolic_physical.cpp',
  'astcenc/astcenc_weight_align.cpp',
  'astcenc/astcenc_weight_quant_xfer_tables.cpp',
  ),
  cpp_args: [
    wrapper_astcenc_simd, '-DASTCENC_SVE=0', '-DASTCENC_SSE=0', '-DASTCENC_AVX=0',
    '-DASTCENC_POPCNT=0', '-DASTCENC_F16C=0', '-w',
  ],
  gnu_symbol_visibility: 'hidden',
  pic: true,
  override_options: ['cpp_std=c++17', 'werror=false'],
)

wrapper_files = files(
  'wrapper_device.c',
  'wrapper_device_memory.c',
  'wrapper_instance.c',
  'graphicsenv_hook.cpp',
  'spirv_patcher.cpp',
  'wrapper_astcenc.cpp',
  'wrapper_bcdec.c',
  'wrapper_bcn_gpu.c',
  'wrapper_log.c',
  'wrapper_physical_device.c',
)

wrapper_deps = [
  idep_vulkan_lite_runtime,
  idep_vulkan_util,
  idep_vulkan_wsi,
  adrenotools_dep,
  dep_android,
]

libvulkan_wrapper = shared_library(
  'vulkan_wrapper',
  [wrapper_entrypoints, wrapper_trampolines, wrapper_features, wrapper_files],
  include_directories: [
    inc_include,
    inc_src,
  ],
  dependencies: [wrapper_deps, vulkan_wsi_deps],
  link_with: libwrapper_astcenc,
  gnu_symbol_visibility: 'hidden',
  install: true,
)
