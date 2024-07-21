
add_library(ax_controller
    LightControllerUI.cpp
    OperatorProgressStatus.cpp
    GUIWindow.cpp
    GUIWindow_slots.cpp
    GUIWindow_nova_bake.cpp
    WorkspaceTracker.cpp
    ExceptionHandlerUI.cpp


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
    ax_renderer
    ax_event
    ax_core

)
# AUTOUIC weirdly doesn't resolve ui includes
set(AUTOGEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ax_controller_autogen/include")
target_include_directories(ax_controller
PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
INTERFACE
        ${AUTOGEN_INCLUDE_DIR}
)

add_subdirectory(cmd)
add_subdirectory(event)
