add_library(ax_processing STATIC
    ImageManager_UV.cpp 
    ImageManager.cpp
    TextureProcessing.cpp
)


target_link_libraries(ax_processing
PRIVATE
        ax_texture
        ax_controller
        ax_core
)

target_include_directories(ax_processing 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(AXOMAE_USE_CUDA)
    target_sources(ax_processing PRIVATE
            cuda/CubemapProcessing.cu
            cuda/texture_generation.cu
    )
    target_link_libraries(ax_processing PRIVATE
            ax_io
            CUDA::cudart_static
    )
    target_compile_definitions(ax_processing PUBLIC AXOMAE_USE_CUDA)
    target_include_directories(ax_processing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
endif()


