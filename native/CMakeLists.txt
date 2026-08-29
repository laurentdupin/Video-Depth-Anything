cmake_minimum_required(VERSION 3.20)
project(video_depth_anything_native VERSION 0.1.0 LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(VDA_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)

if(VDA_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(VDA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${VDA_GENERATED_DIR}")

    function(vda_compile_shader NAME)
        set(SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${VDA_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${VDA_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -O -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_pointwise_gemm_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_tiled_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_implicit_gemm_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_vec4_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=vda_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS
                "${SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(VDA_SHADER_HEADERS
            ${VDA_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half
            bmm_value_half conv2d conv2d8 conv2d_half conv2d8_half
            conv2d_pointwise_gemm conv2d_pointwise_gemm_half
            conv2d8_tiled16x8 conv2d8_tiled16x8_half
            conv2d_implicit_gemm conv2d_implicit_gemm_half
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            elementwise gelu layer_norm linear linear16 linear_half linear16_half
            linear_vec4 linear_vec4_half linear_vec8 linear_vec8_half
            linear_vec16 linear_vec16_half
            prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim
            softmax_lastdim_half temporal_attention
            temporal_attention_stream temporal_geglu
            temporal_group_norm temporal_output temporal_position
            temporal_transpose preprocess_texture resize_depth
            reduce_minmax normalize_depth_image copy_depth_image)
        vda_compile_shader(${SHADER})
    endforeach()
    foreach(SHARED_SHADER quantize_rows_int8 linear_int8_tiled)
        inferbridge_compile_native_support_shader(
            NAME "${SHARED_SHADER}" PREFIX vda
            GENERATED_DIR "${VDA_GENERATED_DIR}"
            EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            OUT_VAR VDA_SHADER_HEADERS)
    endforeach()
endif()

add_library(video_depth_anything_native SHARED
    src/vda.cpp
    src/inferbridge_harness.cpp
    src/model.cpp
    src/encoder_cpu.cpp
    src/temporal_cpu.cpp
    src/dpt_cpu.cpp)
if(VDA_WITH_VULKAN)
    target_sources(video_depth_anything_native PRIVATE
        src/vulkan.cpp
        src/operators.cpp
        src/gpu_model.cpp
        src/gpu_io.cpp
        src/external_gpu.cpp
        src/encoder_gpu.cpp
        src/dpt_gpu.cpp
        src/temporal_gpu.cpp
        ${VDA_SHADER_HEADERS})
    target_include_directories(
        video_depth_anything_native PRIVATE "${VDA_GENERATED_DIR}")
    target_link_libraries(
        video_depth_anything_native PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(
            video_depth_anything_native PRIVATE d3d12 dxgi)
    endif()
    target_compile_definitions(
        video_depth_anything_native PRIVATE VDA_WITH_VULKAN=1)
endif()
target_include_directories(
    video_depth_anything_native
    PUBLIC include
    PRIVATE src)
target_compile_definitions(
    video_depth_anything_native PRIVATE
        VDA_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
set_target_properties(video_depth_anything_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(MSVC)
    target_compile_options(
        video_depth_anything_native PRIVATE /W4 /permissive-)
endif()

add_executable(vda_model_probe tools/model_probe.cpp src/model.cpp)
target_include_directories(vda_model_probe PRIVATE include src)
if(MSVC)
    target_compile_options(vda_model_probe PRIVATE /W4 /permissive-)
endif()

add_executable(vda_temporal_probe
    tools/temporal_probe.cpp
    src/model.cpp
    src/temporal_cpu.cpp)
target_include_directories(vda_temporal_probe PRIVATE include src)
if(MSVC)
    target_compile_options(vda_temporal_probe PRIVATE /W4 /permissive-)
endif()

add_executable(vda_encoder_probe
    tools/encoder_probe.cpp
    src/model.cpp
    src/encoder_cpu.cpp)
target_include_directories(vda_encoder_probe PRIVATE include src)
if(MSVC)
    target_compile_options(vda_encoder_probe PRIVATE /W4 /permissive-)
endif()

add_executable(vda_full_graph_probe
    tools/full_graph_probe.cpp)
target_link_libraries(
    vda_full_graph_probe PRIVATE video_depth_anything_native)
target_include_directories(vda_full_graph_probe PRIVATE include)
if(MSVC)
    target_compile_options(vda_full_graph_probe PRIVATE /W4 /permissive-)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(vda_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        vda_c_abi_smoke PRIVATE video_depth_anything_native)
    target_include_directories(vda_c_abi_smoke PRIVATE include)
    add_test(NAME vda_c_abi_smoke COMMAND vda_c_abi_smoke)
    add_executable(vda_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(
        vda_harness_abi_smoke PRIVATE video_depth_anything_native)
    target_include_directories(
        vda_harness_abi_smoke PRIVATE include)
    add_test(
        NAME vda_harness_abi_smoke COMMAND vda_harness_abi_smoke)
    add_executable(vda_harness_full_graph tests/harness_full_graph.cpp)
    target_link_libraries(
        vda_harness_full_graph PRIVATE video_depth_anything_native)
    target_include_directories(
        vda_harness_full_graph PRIVATE include)
    add_test(
        NAME vda_harness_full_graph COMMAND vda_harness_full_graph)
    set_tests_properties(
        vda_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    if(VDA_WITH_VULKAN)
        if(WIN32)
            add_executable(vda_d3d12_full_graph tests/d3d12_full_graph.cpp)
            target_link_libraries(vda_d3d12_full_graph PRIVATE
                video_depth_anything_native d3d12 dxgi)
            target_include_directories(vda_d3d12_full_graph PRIVATE include)
            add_test(NAME vda_d3d12_full_graph COMMAND vda_d3d12_full_graph)
            set_tests_properties(
                vda_d3d12_full_graph PROPERTIES SKIP_RETURN_CODE 77)
        endif()
        add_executable(vda_vulkan_smoke
            tests/vulkan_smoke.cpp
            src/vulkan.cpp
            src/operators.cpp
            src/gpu_model.cpp
            src/model.cpp
            ${VDA_SHADER_HEADERS})
        target_include_directories(vda_vulkan_smoke PRIVATE
            include src "${VDA_GENERATED_DIR}")
        target_link_libraries(vda_vulkan_smoke PRIVATE Vulkan::Vulkan)
        add_test(NAME vda_vulkan_smoke COMMAND vda_vulkan_smoke)
        add_executable(vda_gpu_encoder_probe
            tools/gpu_encoder_probe.cpp
            src/encoder_gpu.cpp
            src/vulkan.cpp
            src/operators.cpp
            src/gpu_model.cpp
            src/model.cpp
            ${VDA_SHADER_HEADERS})
        target_include_directories(vda_gpu_encoder_probe PRIVATE
            include src "${VDA_GENERATED_DIR}")
        target_link_libraries(
            vda_gpu_encoder_probe PRIVATE Vulkan::Vulkan)
        add_executable(vda_gpu_temporal_probe
            tools/gpu_temporal_probe.cpp
            src/temporal_gpu.cpp
            src/vulkan.cpp
            src/operators.cpp
            src/gpu_model.cpp
            src/model.cpp
            ${VDA_SHADER_HEADERS})
        target_include_directories(vda_gpu_temporal_probe PRIVATE
            include src "${VDA_GENERATED_DIR}")
        target_link_libraries(
            vda_gpu_temporal_probe PRIVATE Vulkan::Vulkan)
        add_executable(
            vda_gpu_full_graph_probe tools/gpu_full_graph_probe.cpp)
        target_link_libraries(
            vda_gpu_full_graph_probe
            PRIVATE video_depth_anything_native)
    endif()
endif()
