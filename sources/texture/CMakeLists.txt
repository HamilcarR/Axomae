add_library(ax_texture 
    EnvmapTextureManager.cpp 
    Texture.cpp 
    TextureGroup.cpp
    texture_utils.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_texture)
target_link_libraries(ax_texture 
PUBLIC  
    ax_io
    ax_database
    ax_renderer
    ax_mesh
    ax_debug
    ax_common
    
)
target_include_directories(ax_texture 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)