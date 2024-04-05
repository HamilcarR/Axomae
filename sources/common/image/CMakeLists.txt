add_library(ax_image 
    Rgb.cpp 
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_image)
target_link_libraries(ax_image 
PRIVATE
    ax_macro
    ax_common
)

target_include_directories(ax_image 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)