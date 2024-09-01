add_library(ax_material STATIC

        Material.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_material)
target_link_libraries(ax_material
PRIVATE
    ax_shader
    ax_texture
    ax_core
    ax_opengl

)
target_include_directories(ax_material 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)