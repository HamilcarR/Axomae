
add_library(ax_core_macro)

target_link_libraries(ax_core PUBLIC ax_core_macro)

target_include_directories(ax_core_macro
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)