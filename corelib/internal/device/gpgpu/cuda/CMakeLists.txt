add_library(ax_core_cuda
        cuda_utils.cpp
        CudaDevice.cpp
        CudaParams.cpp
        CudaDeviceError.cpp
        cuda_buffer_loader.cpp
        cuda_texture_loader.cpp
)

target_link_libraries(ax_core_device PUBLIC ax_core_cuda)

target_link_libraries(ax_core_cuda
PUBLIC
        ax_core_macro
        ax_core_debug
        ax_core_common
        ax_core_thread
        CUDA::cudart_static
)

target_include_directories(ax_core_cuda PRIVATE
    ${AXOMAE_CORELIB_ROOT}

)
add_subdirectory(optix)

if(AXOMAE_USE_CUDA)
    target_compile_definitions(ax_core_cuda PUBLIC AXOMAE_USE_CUDA)
endif ()