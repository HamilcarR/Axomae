set(MEMORY_MODULE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tag_ptrTest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MemoryArenaTest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MemoryPoolAllocatorTest.cpp
)

if(AXOMAE_USE_CUDA)
    list(APPEND MEMORY_MODULE_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tag_device_test.cu
    )

endif()

set(MEMORY_MODULE_SOURCES ${MEMORY_MODULE_SOURCES} PARENT_SCOPE)
