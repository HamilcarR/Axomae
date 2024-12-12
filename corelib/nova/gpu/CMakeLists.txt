autoload_build_utils()

add_library(ax_core_nova_gpu STATIC
        GPURandomGenerator.cpp)

target_link_libraries(ax_core_nova_gpu
PRIVATE
        ax_core_nova_engine
        ax_core_nova_manager
        ax_core_geometry
        ax_core_memory
        ax_core_macro
        ax_core_device
        ax_core_debug
        CUDA::curand
)


target_include_directories(ax_core_nova_gpu
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../

)


if(AXOMAE_USE_CUDA)
    target_compile_definitions(ax_core_nova_gpu PUBLIC AXOMAE_USE_CUDA)
    target_link_libraries(ax_core_nova PRIVATE ax_core_nova_gpu)
endif ()


register_device_compilation(GPURandomGenerator.cpp)