


add_library(ax_core STATIC)

target_include_directories(ax_core
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


add_subdirectory(internal)
add_subdirectory(nova)


