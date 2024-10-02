
add_library(ax_core_device STATIC
        device_resource_interface.h)


target_link_libraries(ax_core PUBLIC ax_core_device)

target_link_libraries(ax_core_device
PRIVATE
        ax_core_macro
        ax_core_debug
)

target_include_directories(ax_core_device
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


add_subdirectory(opengl)
add_subdirectory(vulkan)
add_subdirectory(hip)
add_subdirectory(opencl)

if(AXOMAE_USE_CUDA)
    target_compile_definitions(ax_core_device PUBLIC AXOMAE_USE_CUDA)
    target_sources(ax_core_device PUBLIC DeviceError.cpp)
    set_source_files_properties(device_utils.h PROPERTIES LANGUAGE CUDA)
    add_subdirectory(cuda)
endif()