add_library(ax_shader 
    Shader.cpp
    utils/glsl.cpp
    )

target_link_libraries(${PROJECT_NAME} PUBLIC ax_shader)

target_link_libraries(ax_shader 
PRIVATE
    ax_texture
    ax_opengl
    ax_camera
    ax_scene
    ax_debug

)

target_include_directories(ax_shader 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/utils
    utils
)


#****************************************************************************************************
# Encode shaders as byte arrays
execute_process(COMMAND ${PROJECT_SOURCE_DIR}/scripts/stringify.sh ${PROJECT_SOURCE_DIR} )

