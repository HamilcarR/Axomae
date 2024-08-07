add_library(ax_core_device
        DeviceBufferInterface.h
        DeviceShaderInterface.h
        DeviceTextureInterface.h
        GPUBackendInterface.h
        DeviceError.h
        DeviceParams.h
        DeviceError.cpp
)

target_link_libraries(ax_core PUBLIC ax_core_device)


target_include_directories(ax_core_device
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


add_subdirectory(opengl)
add_subdirectory(cuda)
add_subdirectory(vulkan)
add_subdirectory(hip)
add_subdirectory(opencl)