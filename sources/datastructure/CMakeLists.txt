add_library(ax_datastructure

)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_datastructure)
target_link_libraries(ax_datastructure)




target_include_directories(ax_datastructure
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

