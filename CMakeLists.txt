cmake_minimum_required(VERSION 3.18)
project(Axomae VERSION 1.0 LANGUAGES C CXX)



#****************************************************************************************************
# Encode shaders as byte arrays 
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/stringify.sh ${CMAKE_CURRENT_SOURCE_DIR} )



#****************************************************************************************************
# Compilation 

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# CXX flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(USE_GOLD CACHE BOOL "Use GOLD linker")
set(USE_MOLD CACHE BOOL "USE MOLD linker")

if(USE_GOLD)
    set(CMAKE_CXX_FLAGS "-fuse-ld=gold")
elseif(USE_MOLD)
    set(CMAKE_CXX_FLAGS "-fuse-ld=mold")
endif()

#****************************************************************************************************
# Debug 
set(USE_ASAN "OFF" CACHE BOOL "Enable Address Sanitizer")
if(USE_ASAN)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
endif()


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

#****************************************************************************************************
#Cuda 
set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/cuda_generated)
set(CMAKE_CUDA_FLAGS "--compiler-bindir ${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_ARCHITECTURES "75")
enable_language(CUDA)


#****************************************************************************************************
#Include and paths 

# STB
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stb CACHE PATH "stb_image include path")
if(NOT STB_INCLUDE_PATH)
    message(FATAL_ERROR "STB image path needs to be valid!")
endif()

# Optix
set(OPTIX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/Optix CACHE PATH "Optix include path")

# .ui files paths 
set(UI_FILES_LIST)

#****************************************************************************************************

# Project setup 
qt_standard_project_setup()
set(CMAKE_AUTOUIC OFF)

qt_add_executable(${PROJECT_NAME} sources/main/main.cpp)

# Subdirs include
add_subdirectory(sources)
