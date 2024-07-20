add_library(ax_core_nova_primitive
        NovaGeoPrimitive.cpp
)


target_link_libraries(ax_core_nova_primitive PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_nova_sampler

)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_primitive)

target_include_directories(ax_core_nova_primitive PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
