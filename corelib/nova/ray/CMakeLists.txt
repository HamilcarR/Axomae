add_library(ax_core_nova_ray STATIC
    Ray.cpp
)

target_link_libraries(ax_core_nova_ray PRIVATE
        ax_core_math
        ax_core_macro
        ax_core_memory

)
target_link_libraries(ax_core_nova PUBLIC ax_core_nova_ray)

target_include_directories(ax_core_nova_ray PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
