add_library(ax_nova_camera
)

target_link_libraries(ax_nova PUBLIC ax_nova_camera)
target_link_libraries(ax_nova_camera
PRIVATE
    ax_math

)
target_include_directories(ax_nova_camera
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
