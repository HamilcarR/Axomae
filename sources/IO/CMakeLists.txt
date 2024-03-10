add_library(ax_io  
    ImageImporter.cpp 
    Loader_file.cpp 
    Loader_image.cpp
    Loader_scene.cpp 
    Logger.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_io)
target_link_libraries(ax_io 
PUBLIC
    ax_macro
    ax_scene
    ax_thread
    ax_common
    ax_exception
    ax_editor
    assimp::assimp
    SDL2::SDL2
    SDL2_image::SDL2_image
)
target_include_directories(ax_io 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_INCLUDE_PATH}
)
