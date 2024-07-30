add_library(ax_core_nova
        DrawEngine.cpp
)


target_link_libraries(ax_core PUBLIC ax_core_nova)

target_link_libraries(ax_core_nova
PRIVATE
        ax_core_geometry
        ax_core_memory
        ax_core_thread
        ax_core_macro
        ax_core_device
        ax_core_debug

PUBLIC
        Boost::random
)

target_include_directories(ax_core_nova
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(texturing)
add_subdirectory(shape)
add_subdirectory(scene)
add_subdirectory(engine)
add_subdirectory(ray)
add_subdirectory(primitive)
add_subdirectory(material)
add_subdirectory(integrator)
add_subdirectory(kernel)
add_subdirectory(camera)
add_subdirectory(aggregate)
add_subdirectory(sampler)
add_subdirectory(manager)