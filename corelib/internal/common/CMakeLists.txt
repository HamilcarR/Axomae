
add_library(ax_core_common STATIC)

target_link_libraries(ax_core PUBLIC ax_core_common)
target_link_libraries(ax_core_common
PRIVATE
        ax_core_math
        ax_core_image
        ax_core_exception

PUBLIC
        Boost::core
        ax_core_macro
)

target_include_directories(ax_core_common
PUBLIC
    ${AXOMAE_CORELIB_ROOT}
)
if(AXOMAE_USE_CUDA)
  target_compile_definitions(ax_core_common PUBLIC AXOMAE_USE_CUDA)
endif()
if(AXOMAE_USE_EMBREE)
  target_compile_definitions(ax_core_common PUBLIC AXOMAE_USE_EMBREE)
endif()


add_subdirectory(math)
add_subdirectory(exception)
add_subdirectory(image)

