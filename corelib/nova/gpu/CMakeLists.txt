

add_library(ax_core_nova_gpu
        gpu.cu
)

set_target_properties(ax_core_nova_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ax_core_nova_gpu
PRIVATE
        ax_core_nova_engine
        ax_core_geometry
        ax_core_memory
        ax_core_macro
        ax_core_device
        ax_core_debug
        Boost::random
)


target_link_libraries(ax_core_nova PUBLIC ax_core_nova_gpu)
target_include_directories(ax_core_nova_gpu
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
