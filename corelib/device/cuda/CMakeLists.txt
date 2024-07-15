add_library(ax_core_cuda
)

target_link_libraries(ax_core_device PUBLIC ax_core_cuda)

target_link_libraries(ax_core_cuda
PUBLIC
        ax_core_macro
        CUDA::cudart
)

target_include_directories(ax_core_cuda
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
