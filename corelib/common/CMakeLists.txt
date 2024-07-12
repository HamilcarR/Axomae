
add_library(ax_core_common)

target_link_libraries(ax_core PUBLIC ax_core_common)
target_link_libraries(ax_core_common
PUBLIC
        ax_core_math
        ax_core_image
        ax_core_exception



)

target_include_directories(ax_core_common
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(math)
add_subdirectory(exception)
add_subdirectory(image)

