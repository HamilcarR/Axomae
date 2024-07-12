add_library(ax_opengl 
        CameraFrameBuffer.cpp
        IFrameBuffer.cpp
        GLFrameBuffer.cpp
        GLGeometryBuffer.cpp
        GLRenderBuffer.cpp
        RenderCubeMap.cpp
        RenderQuad.cpp
        FramebufferInterface.h
        GLMutablePixelBufferObject.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_opengl)
target_link_libraries(ax_opengl 

PRIVATE
    ax_mesh
    ax_texture
    ax_renderer
    ax_camera
    ax_shader
    ax_scene
    ax_material
    ax_core
    OpenGL::GL
    OpenGL::GLU
    GLEW::GLEW 

)
target_include_directories(ax_opengl 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(shader)