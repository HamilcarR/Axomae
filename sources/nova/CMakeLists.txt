add_library(ax_nova
        DrawEngine.cpp
        aggregate/bvh/BvhBuilder.cpp

)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_nova)

target_link_libraries(ax_nova
PRIVATE
    ax_math
    ax_macro
    ax_thread
    ax_processing
    ax_geometry
)

target_include_directories(ax_nova
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ray
        ${CMAKE_CURRENT_SOURCE_DIR}/primitive
        ${CMAKE_CURRENT_SOURCE_DIR}/material
        ${CMAKE_CURRENT_SOURCE_DIR}/scene
        ${CMAKE_CURRENT_SOURCE_DIR}/shape
        ${CMAKE_CURRENT_SOURCE_DIR}/aggregate

)

add_subdirectory(texturing)
add_subdirectory(shape)
add_subdirectory(scene)
add_subdirectory(rendering)
add_subdirectory(ray)
add_subdirectory(primitive)
add_subdirectory(material)
add_subdirectory(light)
add_subdirectory(kernel)
add_subdirectory(camera)
add_subdirectory(aggregate)