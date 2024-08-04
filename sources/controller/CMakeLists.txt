add_library(ax_controller STATIC
    LightControllerUI.cpp
    OperatorProgressStatus.cpp
    GUIWindow.cpp
    GUIWindow_slots.cpp
    GUIWindow_nova_bake.cpp
    WorkspaceTracker.cpp
    ExceptionHandlerUI.cpp
    cmd/API.cpp
    cmd/API_initapp.cpp
    cmd/ProgramOptionsManager.cpp
    event/EventController.h
    event/EventInterface.h
)

target_link_libraries(ax_controller
PUBLIC
        ax_config
        ax_io
        ax_database
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
        ax_core
        Boost::program_options
PRIVATE
        SDL2_image::SDL2_image
        SDL2::SDL2
)
# AUTOUIC weirdly doesn't resolve ui includes
set(AUTOGEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ax_controller_autogen/include")
target_include_directories(ax_controller
PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
INTERFACE
        ${AUTOGEN_INCLUDE_DIR}
)

