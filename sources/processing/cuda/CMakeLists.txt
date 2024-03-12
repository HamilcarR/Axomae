add_library(ax_processing_cuda
        CubemapProcessing.cu
        Kernel.cu
)

target_link_libraries(ax_processing PUBLIC ax_processing_cuda)
target_link_libraries(ax_processing_cuda
PUBLIC
        ax_common
        ax_io
        ax_gpu_cuda
        ax_macro
        ax_thread
        ax_math
)

target_include_directories(ax_processing_cuda
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)