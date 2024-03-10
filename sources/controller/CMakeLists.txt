
qt_wrap_ui(UI_FILES_LIST ../editor/Form/main_window.ui)

add_library(ax_controller 
    GUIWindow.cpp
    LightControllerUI.cpp
    OP_ProgressStatus.cpp
    ${UI_FILES_LIST}
    )

target_link_libraries(${PROJECT_NAME} PUBLIC ax_controller)
target_link_libraries(ax_controller 
PUBLIC
    ax_config
    ax_editor 
    ax_processing


    )

target_include_directories(ax_controller 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cmd)

