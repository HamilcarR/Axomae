add_library(ax_cmd 
    api_initapp.cpp 
    API.cpp 
    CmdArgs.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_cmd)
target_link_libraries(ax_cmd 
PRIVATE

    Boost::program_options
    ax_config
    ax_io
    ax_database
    ax_editor
    ax_mesh
    ax_scene
    ax_camera
    ax_material
    ax_shader
    ax_texture
    ax_opengl
    ax_processing
    ax_core

)

target_include_directories(ax_cmd
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}

)
