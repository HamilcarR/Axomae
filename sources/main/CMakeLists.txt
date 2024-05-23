

target_link_libraries(${PROJECT_NAME} 
PUBLIC
        ax_cmd
        SDL2::SDL2
        Boost::program_options
        Boost::stacktrace_addr2line
)

