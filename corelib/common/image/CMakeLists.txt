add_library(ax_core_image
    Rgb.cpp 
)

target_link_libraries(ax_core_common PUBLIC ax_core_image)
target_link_libraries(ax_core_image
PRIVATE
    ax_core_macro
)

target_include_directories(ax_core_image
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)