add_library(ax_nova_texturing
)

target_link_libraries(ax_nova PUBLIC ax_nova_texturing)

target_link_libraries(ax_nova_texturing
PUBLIC
        ax_processing
        ax_macro
)


target_include_directories(ax_nova_texturing
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
