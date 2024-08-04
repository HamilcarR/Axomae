# Encode shaders as byte arrays
set(GEN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/lib/shader/include" )
file(MAKE_DIRECTORY ${GEN_INCLUDE_DIR})

#execute_process(COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_shaders.py ${CMAKE_CURRENT_SOURCE_DIR}/glsl ${GEN_INCLUDE_DIR})
file(GLOB_RECURSE GLSL_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.geom"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.comp"
)
add_custom_target(axcustom_glsl_generate ALL
        COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_shaders.py ${CMAKE_CURRENT_SOURCE_DIR}/glsl ${GEN_INCLUDE_DIR}
        DEPENDS ${GLSL_SRC}
        COMMENT "Generating shaders as byte arrays"
        VERBATIM
)

add_library(ax_shader STATIC
    Shader.cpp
    utils/glsl.cpp
)
add_dependencies(ax_shader axcustom_glsl_generate)

target_link_libraries(ax_shader
PRIVATE
    ax_texture
    ax_opengl
    ax_camera
    ax_scene
    ax_core
)

#****************************************************************************************************





target_include_directories(ax_shader
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GEN_INCLUDE_DIR}
    utils
)

