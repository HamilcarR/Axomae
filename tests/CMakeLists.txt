set(TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})




add_executable(Axomae_test Test.cpp)

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
        ax_controller
        ax_core
        PRIVATE
        compile_opts
)

target_include_directories(Axomae_test
PUBLIC
        ${GTEST_INCLUDE_PATH}
)
add_subdirectory(sources)
add_subdirectory(corelib)

include(GoogleTest)
gtest_discover_tests(Axomae_test)

