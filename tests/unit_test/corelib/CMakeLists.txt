set(NOVA_MODULE_SOURCES "" CACHE INTERNAL "")
set(DATASTRUCTURE_MODULE_SOURCES "" CACHE INTERNAL "" )
set(GEOMETRY_MODULE_SOURCES "" CACHE INTERNAL "" )
set(COMMON_MODULE_SOURCES "" CACHE INTERNAL "")
set(MEMORY_MODULE_SOURCES "" CACHE INTERNAL "")
set(DEVICE_MODULE_SOURCES "" CACHE INTERNAL "")

add_subdirectory(nova)
add_subdirectory(internal/common)
add_subdirectory(internal/datastructure)
add_subdirectory(internal/geometry)
add_subdirectory(internal/memory)
if(AXOMAE_USE_CUDA)
    add_subdirectory(internal/device/gpgpu)
endif()

target_sources(Axomae_test PUBLIC
        ${NOVA_MODULE_SOURCES}
        ${DATASTRUCTURE_MODULE_SOURCES}
        ${GEOMETRY_MODULE_SOURCES}
        ${COMMON_MODULE_SOURCES}
        ${MEMORY_MODULE_SOURCES}
        ${DEVICE_MODULE_SOURCES}
)
