add_library(ax_material Material.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_material)
target_link_libraries(ax_material 
PRIVATE
    ax_shader
    ax_common
    ax_texture
    ax_macro
    ax_thread
    ax_opengl
    ax_math
    ax_debug
    ax_io

)
target_include_directories(ax_material 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)