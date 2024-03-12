add_library(ax_camera Camera.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_camera)
target_link_libraries(ax_camera
PUBLIC
        ax_math
        ax_common
PRIVATE
        ax_scene
)
target_include_directories(ax_camera
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)