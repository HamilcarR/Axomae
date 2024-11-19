add_library(ax_core_memory
        MemoryArena.h
)

target_link_libraries(ax_core PUBLIC ax_core_memory)
target_link_libraries(ax_core_memory
PUBLIC
        ax_core_device
        ax_core_common
        ax_core_thread

)
target_include_directories(ax_core_memory
PUBLIC
        ${AXOMAE_CORELIB_ROOT}
)
