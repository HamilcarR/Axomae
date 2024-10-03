
add_library(ax_core_device STATIC)


target_link_libraries(ax_core PUBLIC ax_core_device)

target_link_libraries(ax_core_device
PRIVATE
        ax_core_macro
        ax_core_debug
)


add_subdirectory(gpgpu)
add_subdirectory(rendering)


