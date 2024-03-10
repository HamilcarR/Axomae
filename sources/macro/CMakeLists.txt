
add_library(ax_macro)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_macro)

target_link_libraries(ax_macro 
PUBLIC
    ax_common
)

target_include_directories(ax_macro 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)