set(PROCESSING_MODULE_SOURCES "" CACHE INTERNAL "")
set(DATABASE_MODULE_SOURCES "" CACHE INTERNAL "")
set(SCENE_MODULE_SOURCES "" CACHE INTERNAL "" )
set(MESH_MODULE_SOURCES "" CACHE INTERNAL "" )
set(CONTROLLER_MODULE_SOURCES "" CACHE INTERNAL "")

add_subdirectory(database)
add_subdirectory(processing)
add_subdirectory(mesh)
add_subdirectory(scene)
add_subdirectory(controller)
add_subdirectory(common)

add_executable(Axomae_test
        Test.cpp
        ${PROCESSING_MODULE_SOURCES}
        ${DATABASE_MODULE_SOURCES}
        ${SCENE_MODULE_SOURCES}
        ${MESH_MODULE_SOURCES}
        ${CONTROLLER_MODULE_SOURCES}
)

target_link_libraries(Axomae_test
PUBLIC
        GTest::gtest_main
        GTest::gtest


PRIVATE
        ax_common
        ax_processing
        ax_texture
        ax_camera
        ax_editor
        ax_scene
        ax_shader
        ax_opengl
        ax_renderer
        ax_mesh
        ax_material
        ax_geometry
        ax_event
)

target_include_directories(Axomae_test
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

gtest_discover_tests(Axomae_test)