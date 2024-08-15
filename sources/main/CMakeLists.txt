

target_link_libraries(${PROJECT_NAME} 
PUBLIC
        ax_controller
        SDL2::SDL2
        Boost::program_options
        Boost::stacktrace_addr2line
)

