add_library(ax_processing 
    ImageManager_UV.cpp 
    ImageManager.cpp 
    OfflineCubemapProcessing.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_processing)
target_link_libraries(ax_processing 
PRIVATE
        ax_macro
        ax_texture
        ax_image
        ax_math
        ax_exception
        ax_geometry
        ax_common
        ax_processing_cuda
        ax_controller
)

target_include_directories(ax_processing 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(cuda)