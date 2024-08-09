set_source_files_properties(gpu.cpp PROPERTIES LANGUAGE CUDA)
add_library(ax_core_nova_kernel gpu.cpp)

target_link_libraries(ax_core_nova_kernel
PRIVATE
        ax_core_geometry
        ax_core_memory
        ax_core_macro
        ax_core_device
        ax_core_debug
PUBLIC
        Boost::random
)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_kernel)
target_include_directories(ax_core_nova_kernel
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
