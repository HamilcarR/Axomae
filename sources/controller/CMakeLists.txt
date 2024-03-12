
add_library(ax_controller
    LightControllerUI.cpp
    OP_ProgressStatus.cpp
    GUIWindow.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_controller)
target_link_libraries(ax_controller
PUBLIC
    ax_config
PRIVATE
    ax_editor
    ax_processing
    ax_mesh
    ax_scene
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_opengl
    ax_geometry
    ax_renderer


)

target_include_directories(ax_controller
PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/sources/editor
)

add_subdirectory(cmd)

