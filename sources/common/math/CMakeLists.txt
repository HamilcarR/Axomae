add_library(ax_math
        math_acos_lt.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC ax_math)
target_link_libraries(ax_math
PUBLIC
        ax_macro
)


target_include_directories(ax_math 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vector
)