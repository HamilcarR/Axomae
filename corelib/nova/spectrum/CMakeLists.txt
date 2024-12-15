autoload_build_utils()



add_library(ax_core_nova_spectrum STATIC Spectrum.cpp)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_spectrum)

target_link_libraries(ax_core_nova_spectrum PRIVATE
        ax_core_device
        ax_core_memory
        ax_core_geometry
        ax_core_math
        ax_core_macro
)


target_include_directories(ax_core_nova_spectrum PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${AXOMAE_CORELIB_ROOT}
)

register_device_compilation(Spectrum.cpp)