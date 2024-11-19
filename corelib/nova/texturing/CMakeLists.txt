add_library(ax_core_nova_texturing STATIC
        nova_texturing.cpp
        NovaTexture.cpp
)
target_link_libraries(ax_core_nova_texturing PRIVATE
    ax_core_macro
    ax_core_math
    ax_core_memory
    ax_core_device
)
target_link_libraries(ax_core_nova PUBLIC ax_core_nova_texturing)

target_include_directories(ax_core_nova_texturing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${AXOMAE_CORELIB_ROOT}
)

if(AXOMAE_USE_CUDA)
    set_source_files_properties(NovaTexture.cpp PROPERTIES LANGUAGE CUDA)
endif()