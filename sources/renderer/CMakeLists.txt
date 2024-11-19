add_library(ax_renderer STATIC
        Drawable.cpp
        light/LightingSystem.cpp
        Renderer.cpp
        RenderPipeline.cpp
        nova/NovaRenderer.cpp
        nova/NovaRenderer_scene.cpp
        nova/NovaRenderer_postprocess.cpp
        nova/NovaRenderer_draw.cpp
        nova/extract_scene.cpp
        nova/nova_bake_image.cpp
        nova/extract_material.cpp
        nova/extract_geometry.cpp
)



target_link_libraries(ax_renderer
PRIVATE 
    Qt6::Gui 
    ax_scene
    ax_camera
    ax_mesh
    ax_opengl
    ax_shader 
    ax_texture
    ax_database
    ax_material
    ax_editor
    ax_processing
    ax_core
    ax_controller
    ax_core_nova

)


target_include_directories(ax_renderer 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/light

)

