add_library(ax_scene 
    Node.cpp 
    Scene.cpp 
    SceneHierarchy.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_scene)
target_link_libraries(ax_scene 
PUBLIC
        ax_common
        ax_math
        ax_datastructure
PRIVATE
        ax_material
        ax_mesh
        ax_camera
        ax_database
        ax_shader
        ax_texture
        ax_thread
        ax_opengl
        ax_geometry
        ax_renderer
        ax_event
        ax_nova





)
target_include_directories(ax_scene 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 

    
)
