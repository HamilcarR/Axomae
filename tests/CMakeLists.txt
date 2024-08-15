#1) add source path variable to test module
set(PROCESSING_MODULE_SOURCES "" CACHE INTERNAL "")
set(DATABASE_MODULE_SOURCES "" CACHE INTERNAL "")
set(SCENE_MODULE_SOURCES "" CACHE INTERNAL "" )
set(MESH_MODULE_SOURCES "" CACHE INTERNAL "" )
set(CONTROLLER_MODULE_SOURCES "" CACHE INTERNAL "")
set(NOVA_MODULE_SOURCES "" CACHE INTERNAL "")
set(DATASTRUCTURE_MODULE_SOURCES "" CACHE INTERNAL "" )
set(GEOMETRY_MODULE_SOURCES "" CACHE INTERNAL "" )
set(TEXTURE_MODULE_SOURCES "" CACHE INTERNAL "")
set(COMMON_MODULE_SOURCES "" CACHE INTERNAL "")
set(MEMORY_MODULE_SOURCES "" CACHE INTERNAL "")

#2) add subdir
add_subdirectory(database)
add_subdirectory(processing)
add_subdirectory(mesh)
add_subdirectory(scene)
add_subdirectory(controller)
add_subdirectory(common)
add_subdirectory(nova)
add_subdirectory(datastructure)
add_subdirectory(geometry)
add_subdirectory(texture)
add_subdirectory(memory)

#3) link sources
add_executable(Axomae_test
        Test.cpp
        ${PROCESSING_MODULE_SOURCES}
        ${DATABASE_MODULE_SOURCES}
        ${SCENE_MODULE_SOURCES}
        ${MESH_MODULE_SOURCES}
        ${CONTROLLER_MODULE_SOURCES}
        ${NOVA_MODULE_SOURCES}
        ${DATASTRUCTURE_MODULE_SOURCES}
        ${GEOMETRY_MODULE_SOURCES}
        ${TEXTURE_MODULE_SOURCES}
        ${COMMON_MODULE_SOURCES}
        ${MEMORY_MODULE_SOURCES}
)

#4) link lib
target_link_libraries(Axomae_test
PUBLIC
        gtest
        gtest_main
PRIVATE
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
        ax_core
)

target_include_directories(Axomae_test
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GTEST_INCLUDE_PATH}
)

include(GoogleTest)
gtest_discover_tests(Axomae_test)

