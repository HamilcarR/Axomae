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

set(GEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ax_shader_autogen/include" )
#****************************************************************************************************
# Encode shaders as byte arrays
execute_process(COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_shaders.py ${CMAKE_CURRENT_SOURCE_DIR}/glsl ${GEN_INCLUDE_DIR})


target_include_directories(ax_shader
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GEN_INCLUDE_DIR}
    utils
)

