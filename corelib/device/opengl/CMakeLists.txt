add_library(ax_core_opengl
        GLRenderBuffer.cpp
        GLMutablePixelBufferObject.cpp
        GLFrameBuffer.cpp
        GLVertexBufferObject.cpp
        GLVertexArrayObject.cpp
        GLIndexBufferObject.cpp
)

target_link_libraries(ax_core_device PUBLIC ax_core_opengl)

target_link_libraries(ax_core_opengl
PRIVATE
        ax_core_debug
        ax_core_math
        ax_core_macro
        ax_core_thread
        ax_core_common

PUBLIC
        OpenGL::GL
        OpenGL::GLU
        GLEW::GLEW
)


target_include_directories(ax_core_opengl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)