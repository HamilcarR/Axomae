add_library(ax_renderer 
    Drawable.cpp 
    LightingSystem.cpp 
    Renderer.cpp 
    RenderPipeline.cpp
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
)


target_include_directories(ax_renderer 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

