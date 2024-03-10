add_library(ax_cmd 
    api_initapp.cpp 
    API.cpp 
    CmdArgs.cpp
)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_cmd)
target_link_libraries(ax_cmd 
PUBLIC 
    ax_config
    Boost::program_options
    ax_io
)

target_include_directories(ax_cmd
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)