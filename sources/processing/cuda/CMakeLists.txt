add_library(ax_processing_cuda
        CubemapProcessing.cu
        Kernel.cu
)

target_link_libraries(ax_processing PUBLIC ax_processing_cuda)
gpu_separable_compilation(ax_processing_cuda)
target_link_libraries(ax_processing_cuda
PRIVATE
        ax_controller
        ax_core
        ax_io

)

target_include_directories(ax_processing_cuda
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)