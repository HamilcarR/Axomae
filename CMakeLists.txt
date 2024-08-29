if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source build not allowed.")
endif()

cmake_minimum_required(VERSION 3.27)
project(Axomae VERSION 1.0 LANGUAGES C CXX CUDA )


# Custom variables
set(AXOMAE_UI_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/sources/editor/Form")
#****************************************************************************************************
# Compilation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CXX flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(AXOMAE_USE_GOLD "ON" CACHE BOOL "Use GOLD linker")
set(AXOMAE_USE_MOLD "OFF" CACHE BOOL "USE MOLD linker")

# CUDA Compilation
set(AXOMAE_USE_CUDA "OFF" CACHE BOOL "Use GPU when available")

#****************************************************************************************************
#custom project variables
set(AXOMAE_STATS_TIMER "OFF" CACHE BOOL "Enable performance loggers . Note : Displays only in --verbose mode")
option(AXOMAE_LUT_HIGHP "Enable high precision when computing lookup tables." OFF)
option(AXOMAE_LUT_MEDIUMP "Enable medium precision when computing lookup tables." ON)
option(AXOMAE_LUT_LOWP "Enable low precision when computing lookup tables." OFF)
option(AXOMAE_BUILD_TESTS "Build unit tests." OFF)


#****************************************************************************************************
# Dependencies
# Command line utilities

# GUI
find_package(Qt6 COMPONENTS Gui REQUIRED)
find_package(Qt6 COMPONENTS OpenGLWidgets REQUIRED)
find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
# GPGPU / real time render
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit)

if (AXOMAE_USE_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "Found CUDA toolkit.")
    message(STATUS "Enabling NVCC Separable Compilation")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "--compiler-bindir ${CMAKE_CXX_COMPILER} --expt-relaxed-constexpr")
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/cuda_generated)
endif ()

#****************************************************************************************************
#Include and paths

# UI files path
set(CMAKE_AUTOUIC_SEARCH_PATHS ${AXOMAE_UI_LOCATION})

# Cache
# Assimp
set(ASSIMP_INCLUDE_PATH "" CACHE INTERNAL "")

# STB
set(STB_INCLUDE_PATH "" CACHE INTERNAL "")

# GLM
set(GLM_INCLUDE_PATH "" CACHE INTERNAL "")

# Optix
set(OPTIX_INCLUDE_PATH "" CACHE INTERNAL "")

# Boost
set(BOOST_INCLUDE_PATH "" CACHE INTERNAL "")

# OpenEXR
set(OPENEXR_INCLUDE_PATH "" CACHE INTERNAL "")

# GTest
set(GTEST_INCLUDE_PATH "" CACHE INTERNAL "")

add_subdirectory(vendor)
#****************************************************************************************************
# Project setup
qt_standard_project_setup()



qt_add_executable(${PROJECT_NAME} sources/main/main.cpp)

add_subdirectory(sources)
add_subdirectory(corelib)

#****************************************************************************************************
# Link
if(AXOMAE_USE_GOLD)
    target_link_options(${PROJECT_NAME} PUBLIC -fuse-ld=gold)
elseif(AXOMAE_USE_MOLD)
    target_link_options(${PROJECT_NAME} PUBLIC -fuse-ld=mold)
endif()

#****************************************************************************************************
# Debug
set(AXOMAE_USE_ASAN "OFF" CACHE BOOL "Enable Address Sanitization")
if(AXOMAE_USE_ASAN)
    target_compile_options(${PROJECT_NAME} PUBLIC
            -fno-omit-frame-pointer
            -fsanitize=address
    )
    target_link_options(${PROJECT_NAME} PUBLIC
            -fno-omit-frame-pointer
            -fsanitize=address
    )
endif()

set(AXOMAE_USE_TSAN "OFF" CACHE BOOL "Enable thread Sanitization")
if(AXOMAE_USE_TSAN)
    target_compile_options(${PROJECT_NAME} PUBLIC
            -fsanitize=thread
    )
    target_link_options(${PROJECT_NAME} PUBLIC
            -fsanitize=thread
    )
endif()

#****************************************************************************************************
# Testing

if(AXOMAE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
