add_library(ax_processing 
    ImageManager_UV.cpp 
    ImageManager.cpp 
    OfflineCubemapProcessing.cpp
)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_processing)
target_link_libraries(ax_processing 
PUBLIC
    ax_exception
    ax_common 
PRIVATE
    ax_processing_cuda
    ax_gpu_cuda
)

target_include_directories(ax_processing 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cuda)