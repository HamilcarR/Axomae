add_library(ax_event
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_event)

target_link_libraries(ax_event
PRIVATE
    ax_core
)

target_include_directories(ax_event
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)