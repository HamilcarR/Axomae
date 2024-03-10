add_library(ax_shader 
    Shader.cpp
    utils/glsl.cpp
    )

target_link_libraries(${PROJECT_NAME} PUBLIC ax_shader)

target_link_libraries(ax_shader 
PUBLIC 
    ax_texture
)

target_include_directories(ax_shader 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    utils
)