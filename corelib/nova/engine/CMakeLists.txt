
add_library(ax_core_nova_engine STATIC
        nova_engine.cpp
        nova_exception.cpp
)

target_link_libraries(ax_core_nova_engine PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_device
        ax_core_memory
)
target_link_libraries(ax_core_nova PUBLIC ax_core_nova_engine)

target_include_directories(ax_core_nova_engine PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)

if(AXOMAE_USE_CUDA)
    set_source_files_properties(nova_exception.cpp PROPERTIES LANGUAGE CUDA)
    endif()