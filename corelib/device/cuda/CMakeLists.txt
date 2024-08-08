add_library(ax_core_cuda
        cuda_utils.cpp
        CudaDevice.cpp
        CudaDevice.h
        CudaParams.cpp
        CudaParams.h
        kernel/kernel_launcher.cuh
        kernel/kernel_launcher.cu
)

target_link_libraries(ax_core_device PUBLIC ax_core_cuda)

target_link_libraries(ax_core_cuda
PUBLIC
        ax_core_macro
        ax_core_debug
        ax_core_common
        ax_core_thread
        CUDA::cudart
)

target_include_directories(ax_core_cuda
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(optix)

if(AXOMAE_USE_CUDA)
    target_compile_definitions(ax_core_cuda PUBLIC AXOMAE_USE_CUDA)
endif (AXOMAE_USE_CUDA)