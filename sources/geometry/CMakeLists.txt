add_library(ax_geometry
        BoundingBox.cpp
        Object3D.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_geometry)
target_link_libraries(ax_geometry
PRIVATE
        ax_math
)

target_include_directories(ax_geometry 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)