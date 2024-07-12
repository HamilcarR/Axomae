add_library(ax_core_geometry
        BoundingBox.cpp
        Object3D.cpp
)

target_link_libraries(ax_core PUBLIC ax_core_geometry)
target_link_libraries(ax_core_geometry
PRIVATE
        ax_core_common
)

target_include_directories(ax_core_geometry
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)