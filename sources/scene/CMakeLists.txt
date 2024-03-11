add_library(ax_scene 
    Node.cpp 
    Scene.cpp 
    SceneHierarchy.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_scene)
target_link_libraries(ax_scene 
PRIVATE 
    ax_material
    ax_common
    ax_math
)
target_include_directories(ax_scene 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 

    
)
