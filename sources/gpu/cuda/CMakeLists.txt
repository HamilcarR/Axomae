add_library(ax_gpu_cuda)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_gpu_cuda)
target_link_libraries(ax_gpu_cuda
PUBLIC
    CUDA::cudart
PRIVATE
    ax_io
)



target_include_directories(ax_gpu_cuda
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

)