add_library(ax_config STATIC
    Config.cpp
)

target_link_libraries(ax_config
PUBLIC
    ax_io
    ax_core
)


target_include_directories(ax_config 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

)


