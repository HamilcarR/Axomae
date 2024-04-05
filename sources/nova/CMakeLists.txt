add_library(ax_nova
        Ray.cpp
)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_nova)

target_link_libraries(ax_nova
PRIVATE
    ax_math
    ax_macro
)

target_include_directories(ax_nova
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)