
qt_wrap_ui(UI_FILES_LIST Form/texture_viewer.ui)
qt_wrap_ui(UI_FILES_LIST Form/main_window.ui)
add_library(ax_editor 
    ProgressStatusWidget.cpp 
    TextureViewerWidget.cpp
    renderer/GLViewer.cpp 
    renderer/HdrListView.cpp
    image/Thumbnail.cpp
    UV/MeshListView.cpp 
    UV/SceneSelector.cpp
    ${UI_FILES_LIST}
)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_editor)
target_link_libraries(ax_editor 
PRIVATE
    Qt6::Widgets
    Qt6::OpenGLWidgets
    ax_common
    ax_database
    ax_macro
    ax_image
)

target_include_directories(ax_editor 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    Form
    image
    renderer
    UV 
)

