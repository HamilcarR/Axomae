add_library(ax_material Material.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_material)
target_link_libraries(ax_material 
PUBLIC 
    ax_shader
    ax_common
    ax_texture
)
target_include_directories(ax_material 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
    ../gpu/opengl/shader
    ../macro
)