add_library(ax_nova
        ray/Ray.cpp
        texturing/nova_texturing.cpp
        primitive/shape/Sphere.cpp
        DrawEngine.cpp

)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_nova)

target_link_libraries(ax_nova
PRIVATE
    ax_math
    ax_macro
    ax_thread
    ax_processing
)

target_include_directories(ax_nova
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ray
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive

)
