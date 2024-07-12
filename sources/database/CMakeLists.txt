add_library(ax_database 
    ImageDatabase.cpp 
    INodeDatabase.cpp 
    LightingDatabase.cpp 
    ResourceDatabaseManager.cpp 
    ShaderDatabase.cpp 
    TextureDatabase.cpp
    EnvmapTextureManager.cpp
    model/ImageModel.cpp
        QueryResult.h
        DatabaseStorage.h
        DatabaseInterface.h
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_database)
target_link_libraries(ax_database 
PRIVATE
    ax_scene
    ax_editor
    ax_texture
    ax_shader
    ax_renderer
    ax_opengl
    ax_camera
    ax_controller
    ax_mesh
    ax_material
    ax_core


)

target_include_directories(ax_database 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    model
)

