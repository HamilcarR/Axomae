


add_library(ax_core SHARED)

target_include_directories(ax_core
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(thread)
add_subdirectory(common)
add_subdirectory(macro)
add_subdirectory(nova)
add_subdirectory(datastructure)
add_subdirectory(device)
add_subdirectory(geometry)
add_subdirectory(debug)
add_subdirectory(memory)


