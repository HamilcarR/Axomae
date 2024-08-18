add_library(ax_config 
    Config.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_config)
gpu_separable_compilation(ax_config)
target_link_libraries(ax_config
PUBLIC
    ax_io
    ax_core
)


target_include_directories(ax_config 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

)


