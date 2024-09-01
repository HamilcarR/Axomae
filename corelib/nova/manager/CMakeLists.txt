add_library(ax_core_nova_manager STATIC
    NovaResourceManager.cpp
)

target_link_libraries(ax_core_nova_manager PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_nova_sampler
        ax_core_memory
        ax_core_device
)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_manager)

target_include_directories(ax_core_nova_manager PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)