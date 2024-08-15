add_library(ax_core_nova_scene
    nova_scene.cpp
)

target_link_libraries(ax_core_nova_scene PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_nova_sampler
        ax_core_memory
        ax_core_device

)
target_link_libraries(ax_core_nova PUBLIC ax_core_nova_scene)

target_include_directories(ax_core_nova_scene PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
