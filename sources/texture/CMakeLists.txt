add_library(ax_texture 
    Texture.cpp 
    TextureGroup.cpp
    texture_utils.cpp
    GenericTextureProcessing.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_texture)
target_link_libraries(ax_texture 
PRIVATE
    ax_io
    ax_renderer
    ax_shader
    ax_core
    ax_opengl
PUBLIC
    ax_controller
    ax_database

)
target_include_directories(ax_texture 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)