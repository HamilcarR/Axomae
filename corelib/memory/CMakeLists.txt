add_library(ax_core_memory
        Mempool.cpp
        Mempool.h)

target_link_libraries(ax_core PUBLIC ax_core_memory)

target_include_directories(ax_core_memory
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)