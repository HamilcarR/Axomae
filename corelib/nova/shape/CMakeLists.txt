add_library(ax_core_nova_shape
        Sphere.cpp
        Square.cpp
        Box.cpp
        Triangle.cpp
)

target_link_libraries(ax_core_nova_shape PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro


)
target_link_libraries(ax_core_nova PUBLIC ax_core_nova_shape)

target_include_directories(ax_core_nova_shape PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
