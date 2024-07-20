add_library(ax_core_nova_sampler
        Sampler.cpp
)

target_link_libraries(ax_core_nova_sampler PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
PUBLIC
        ax_core_cuda
        Boost::random
)
target_link_libraries(ax_core_nova PUBLIC ax_core_nova_sampler)

target_include_directories(ax_core_nova_sampler PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)
