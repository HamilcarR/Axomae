add_library(ax_mesh STATIC
    Mesh.cpp
)


target_link_libraries(ax_mesh
PRIVATE
    ax_scene
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_core
    ax_controller
    ax_opengl
    ax_database

)
target_include_directories(ax_mesh 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
