add_library(ax_editor
        ProgressStatusWidget.cpp
        TextureViewerWidget.cpp
        renderer/GLViewer.cpp
        renderer/HdrListView.cpp
        renderer/SceneListView.cpp
        image/Thumbnail.cpp
        UV/MeshListView.cpp
        UV/SceneSelector.cpp

)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_editor)
target_link_libraries(ax_editor 
PUBLIC
    Qt6::Widgets
    Qt6::OpenGLWidgets
    ax_common
    ax_database
    ax_macro
    ax_image
    ax_math
    ax_io
    ax_config


PRIVATE
    ax_mesh
    ax_scene
    ax_thread
    ax_controller
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_opengl
    ax_geometry
    ax_renderer
    ax_debug
    ax_event
)

set(AUTOGEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ax_editor_autogen/include")
target_include_directories(ax_editor 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    Form
    image
    renderer
    UV
INTERFACE
    ${AUTOGEN_INCLUDE_DIR}

)


