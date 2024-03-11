add_library(ax_texture 
    Texture.cpp 
    TextureGroup.cpp
    texture_utils.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_texture)
target_link_libraries(ax_texture 
PRIVATE
    ax_io
    ax_database
    ax_renderer
    ax_debug
    ax_common
    ax_math
    ax_shader
    ax_thread
    ax_macro
    ax_controller
    ax_opengl
)
target_include_directories(ax_texture 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)