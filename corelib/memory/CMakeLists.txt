add_library(ax_core_memory
        Mempool.cpp
        Mempool.h
        Tptr.cpp
        Tptr.h)

target_link_libraries(ax_core PUBLIC ax_core_memory)
target_link_libraries(ax_core_memory PRIVATE
    ax_core_device

)
target_include_directories(ax_core_memory
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)