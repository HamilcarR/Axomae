add_library(ax_texture 
    EnvmapTextureManager.cpp 
    Texture.cpp 
    TextureGroup.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_texture)
target_link_libraries(ax_texture 
PUBLIC  
    ax_io
    ax_database
    ax_renderer
    ax_mesh
    ax_debug
    
)
target_include_directories(ax_texture 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)