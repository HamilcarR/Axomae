


add_library(ax_controller 
    GUIWindow.cpp
    LightControllerUI.cpp
    OP_ProgressStatus.cpp
    )

target_link_libraries(${PROJECT_NAME} PUBLIC ax_controller)
target_link_libraries(ax_controller 
PRIVATE
    ax_config
    ax_editor 
    ax_processing
    )

target_include_directories(ax_controller 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cmd)

