cmake_minimum_required(VERSION 3.20)
project(video_depth_anything_native VERSION 0.1.0 LANGUAGES C CXX)

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

add_library(video_depth_anything_native SHARED src/vda.cpp)
target_include_directories(
    video_depth_anything_native PUBLIC include)
target_compile_definitions(
    video_depth_anything_native PRIVATE VDA_BUILD_DLL)
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

include(CTest)
if(BUILD_TESTING)
    add_executable(vda_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        vda_c_abi_smoke PRIVATE video_depth_anything_native)
    target_include_directories(vda_c_abi_smoke PRIVATE include)
    add_test(NAME vda_c_abi_smoke COMMAND vda_c_abi_smoke)
endif()
