add_library(ax_editor STATIC
        ProgressStatusWidget.cpp
        TextureViewerWidget.cpp
        MessageBoxWidget.cpp
        renderer/GLViewer.cpp
        renderer/HdrListView.cpp
        renderer/SceneListView.cpp
        image/Thumbnail.cpp
        UV/MeshListView.cpp
        UV/SceneSelector.cpp
        nova/NovaViewerWidget.cpp
        metadata/RgbDisplayerLabel.cpp
        ContextMenuWidget.cpp
)



target_link_libraries(ax_editor
PUBLIC
    Qt6::Widgets
    Qt6::OpenGLWidgets
    ax_database
    ax_io
    ax_config

PRIVATE
    ax_mesh
    ax_scene
    ax_core
    ax_controller
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_opengl
    ax_renderer
)

set(AUTOGEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ax_editor_autogen/include")
target_include_directories(ax_editor 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    Form
    image
    renderer
    UV
    nova
INTERFACE
    ${AUTOGEN_INCLUDE_DIR}

)


