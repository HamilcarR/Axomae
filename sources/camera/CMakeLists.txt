add_library(ax_camera Camera.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_camera)
target_link_libraries(ax_camera
PUBLIC
        ax_math
        ax_common
        ax_macro
PRIVATE
        ax_scene
        ax_event
        ax_io
)
target_include_directories(ax_camera
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)