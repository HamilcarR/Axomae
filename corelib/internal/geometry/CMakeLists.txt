autoload_build_utils()

add_library(ax_core_geometry STATIC
        BoundingBox.cpp
        Object3D.cpp
)

target_link_libraries(ax_core PUBLIC ax_core_geometry)
target_link_libraries(ax_core_geometry
PRIVATE
        ax_core_common
        ax_core_macro
)

target_include_directories(ax_core_geometry
PUBLIC
    ${AXOMAE_CORELIB_ROOT}
)

register_device_compilation(BoundingBox.cpp)
