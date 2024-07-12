add_library(ax_processing 
    ImageManager_UV.cpp 
    ImageManager.cpp
    TextureProcessing.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_processing)
target_link_libraries(ax_processing 
PRIVATE
        ax_texture
        ax_processing_cuda
        ax_controller
        ax_core
)

target_include_directories(ax_processing 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cuda)