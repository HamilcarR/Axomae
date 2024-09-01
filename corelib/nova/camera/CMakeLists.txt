add_library(ax_core_nova_camera STATIC
    nova_camera.cpp
)


target_link_libraries(ax_core_nova_camera PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_memory
)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_camera)

target_include_directories(ax_core_nova_camera PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)