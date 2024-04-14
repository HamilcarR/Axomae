add_library(ax_renderer
        Drawable.cpp
        light/LightingSystem.cpp
        Renderer.cpp
        RenderPipeline.cpp
        NovaRenderer.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_renderer)
target_link_libraries(ax_renderer 
PRIVATE 
    Qt6::Gui 
    ax_scene
    ax_camera
    ax_mesh
    ax_opengl
    ax_shader 
    ax_geometry
    ax_macro
    ax_texture
    ax_database
    ax_material
    ax_debug
    ax_editor
    ax_event
    ax_processing


)


target_include_directories(ax_renderer 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/light
)

