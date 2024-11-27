add_library(ax_io STATIC
    ImageImporter.cpp 
    Loader_file.cpp 
    Loader_image.cpp
    Loader_scene.cpp
    Loader_material.cpp
    Loader_geometry.cpp
)

target_link_libraries(ax_io
PRIVATE
    ax_controller
    ax_editor
    ax_scene
    ax_database
    ax_mesh
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_opengl
    ax_core

    OpenEXR::OpenEXR
    assimp
    SDL2::SDL2
    SDL2_image::SDL2_image
)

target_include_directories(ax_io
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_INCLUDE_PATH}
    ${ASSIMP_INCLUDE_PATH}
    ${OpenEXR_INCLUDE_DIRS}
)
