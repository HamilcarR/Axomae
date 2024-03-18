add_library(ax_math
        Vector.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_math)
target_link_libraries(ax_math 
)


target_include_directories(ax_math 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)