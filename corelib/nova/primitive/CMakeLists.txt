autoload_build_utils()

add_library(ax_core_nova_primitive STATIC
        NovaGeoPrimitive.cpp
        PrimitiveInterface.cpp
)


target_link_libraries(ax_core_nova_primitive PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_nova_sampler
        ax_core_nova_shape
        ax_core_device
        ax_core_memory

)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_primitive)

target_include_directories(ax_core_nova_primitive PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${AXOMAE_CORELIB_ROOT}
)

register_device_compilation(PrimitiveInterface.cpp NovaGeoPrimitive.cpp)