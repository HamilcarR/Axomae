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
    ax_thread
    ax_common
    ax_exception
    ax_controller
    ax_math


PRIVATE
    ax_editor
    ax_image
    ax_scene
    ax_database
    ax_mesh
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_opengl
    ax_geometry


    assimp
    SDL2::SDL2
    SDL2_image::SDL2_image

)


target_include_directories(ax_io
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_INCLUDE_PATH}
    ${ASSIMP_INCLUDE_PATH}
)