add_library(ax_camera
        Camera.cpp
        ArcballCamera.cpp
        FreePerspectiveCamera.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_camera)
target_link_libraries(ax_camera
PUBLIC
        ax_event
PRIVATE
        ax_scene
        ax_io
        ax_core
)
target_include_directories(ax_camera
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)