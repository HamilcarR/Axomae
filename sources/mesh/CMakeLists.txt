add_library(ax_mesh 
    BoundingBox.cpp
    Mesh.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_mesh)
target_link_libraries(ax_mesh 
PUBLIC 
    ax_scene
)
target_include_directories(ax_mesh 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

    )
