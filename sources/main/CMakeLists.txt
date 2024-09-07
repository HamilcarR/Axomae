

target_link_libraries(${PROJECT_NAME} PUBLIC
        ax_core
        ax_controller
        ax_camera
        ax_config
        ax_database
        ax_editor
        ax_opengl
        ax_io
        ax_material
        ax_mesh
        ax_processing
        ax_renderer
        ax_scene
        ax_texture

        SDL2::SDL2
        Boost::program_options
        Boost::stacktrace_addr2line
)




