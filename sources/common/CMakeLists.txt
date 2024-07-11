
add_library(ax_common)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_common)
target_link_libraries(ax_common 
PRIVATE 
    ax_math
    ax_image
    ax_exception
)

target_include_directories(ax_common 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(math)
add_subdirectory(exception)
add_subdirectory(image)

