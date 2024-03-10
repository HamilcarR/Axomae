add_library(ax_opengl 
    CameraFrameBuffer.cpp 
    FrameBufferInterface.cpp 
    GLFrameBuffer.cpp 
    GLGeometryBuffer.cpp 
    GLRenderBuffer.cpp 
    RenderCubeMap.cpp 
    RenderQuad.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_opengl)
target_link_libraries(ax_opengl 
PUBLIC
    ax_mesh
    ax_geometry
    OpenGL::GL
    OpenGL::GLU
    GLEW::GLEW 

)
target_include_directories(ax_opengl 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(shader)