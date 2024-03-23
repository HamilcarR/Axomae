cmake_minimum_required(VERSION 3.27)
project(Axomae VERSION 1.0 LANGUAGES C CXX)

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

set(USE_GOLD "ON" CACHE BOOL "Use GOLD linker")
set(USE_MOLD "OFF" CACHE BOOL "USE MOLD linker")

# CUDA Compilation
set(CMAKE_CUDA_FLAGS "--compiler-bindir ${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_ARCHITECTURES "75")
enable_language(CUDA)
#****************************************************************************************************
# Debug 
set(USE_ASAN "OFF" CACHE BOOL "Enable Address Sanitizer")
if(USE_ASAN)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")

endif()

set(USE_STATS_TIMER "OFF" CACHE BOOL "Enable performance loggers . Note : Displays only in --verbose mode")

#****************************************************************************************************
# Dependencies 

# Command line utilities
find_package(Boost COMPONENTS program_options REQUIRED)
# GUI
find_package(Qt6 COMPONENTS Gui REQUIRED)
find_package(Qt6 COMPONENTS OpenGLWidgets REQUIRED)
find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
# GPGPU / real time render
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
# Model imports 
find_package(assimp REQUIRED)
# Unit-Tests
find_package(GTest REQUIRED)
#****************************************************************************************************
#Include and paths
#Cuda
set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/cuda_generated)
# UI files path
set(CMAKE_AUTOUIC_SEARCH_PATHS ${AXOMAE_UI_LOCATION})


# Cache
# STB
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stb CACHE PATH "stb_image include path")
if(NOT STB_INCLUDE_PATH)
    message(FATAL_ERROR "STB image path needs to be valid!")
endif()
# Optix
set(OPTIX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/Optix CACHE PATH "Optix include path")
#****************************************************************************************************
# Project setup
qt_standard_project_setup()
qt_add_executable(${PROJECT_NAME} sources/main/main.cpp)

add_subdirectory(sources)
#****************************************************************************************************
# Testing
option(BUILD_TESTS "Build unit tests for the project" ON)
if(BUILD_TESTS)
    enable_testing()
    # Subdirs include
    add_subdirectory(tests)
endif()
#****************************************************************************************************
# Link
if(USE_GOLD)
    target_link_options(${PROJECT_NAME} PUBLIC -fuse-ld=gold)
elseif(USE_MOLD)
    target_link_options(${PROJECT_NAME} PUBLIC -fuse-ld=mold)
endif()


