add_library(ax_core_worker
)

target_link_libraries(ax_core_thread PUBLIC ax_core_worker)


target_include_directories(ax_core_worker
PUBLIC
        ${AXOMAE_CORELIB_ROOT}
)