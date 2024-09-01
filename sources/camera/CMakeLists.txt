add_library(ax_camera STATIC
        Camera.cpp
        ArcballCamera.cpp
        FreePerspectiveCamera.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_camera)
target_link_libraries(ax_camera
PUBLIC
        ax_controller
PRIVATE
        ax_scene
        ax_core
)
target_include_directories(ax_camera
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)