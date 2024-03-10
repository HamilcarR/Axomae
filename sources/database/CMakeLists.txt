add_library(ax_database 
    ImageDatabase.cpp 
    INodeDatabase.cpp 
    LightingDatabase.cpp 
    ResourceDatabaseManager.cpp 
    ShaderDatabase.cpp 
    TextureDatabase.cpp
    model/ImageModel.cpp
    )

target_link_libraries(${PROJECT_NAME} PUBLIC ax_database)
target_link_libraries(ax_database 
PUBLIC
    ax_macro  
    ax_scene
    ax_editor
    ax_common 
)

target_include_directories(ax_database 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    model
)


