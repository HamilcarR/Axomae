add_library(ax_opengl STATIC
        CameraFrameBuffer.cpp
        FramebufferHelper.cpp
        PackedGLGeometryBuffer.cpp
        RenderCubeMap.cpp
        RenderQuad.cpp
)

target_link_libraries(ax_opengl
PRIVATE
        ax_texture
        ax_core
        ax_controller
        ax_mesh
        ax_database
        ax_renderer
        ax_camera
        ax_shader
        ax_scene
        ax_material

)

target_include_directories(ax_opengl
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(shader)