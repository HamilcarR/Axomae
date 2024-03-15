add_library(ax_event
        EventController.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_event)

target_link_libraries(ax_event
PRIVATE

)

target_include_directories(ax_event
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)