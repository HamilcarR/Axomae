add_library(ax_scene STATIC
    Node.cpp 
    Scene.cpp 
    SceneHierarchy.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_scene)

gpu_separable_compilation(ax_scene)
target_link_libraries(ax_scene
PRIVATE
        ax_material
        ax_mesh
        ax_camera
        ax_database
        ax_shader
        ax_texture
        ax_opengl
        ax_renderer
        ax_controller
PUBLIC
        ax_core_nova_shape



)
target_include_directories(ax_scene 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 

    
)
