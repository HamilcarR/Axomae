

add_library(ax_core_nova_gpu STATIC
        gpu.cu
)

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


target_include_directories(ax_core_nova_gpu
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_compile_definitions(ax_core_nova_gpu PUBLIC AXOMAE_USE_CUDA)
if(AXOMAE_USE_CUDA)
    target_link_libraries(ax_core_nova PRIVATE ax_core_nova_gpu)
endif ()