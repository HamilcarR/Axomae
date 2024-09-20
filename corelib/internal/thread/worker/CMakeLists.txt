add_library(ax_core_worker
)

target_link_libraries(ax_core_thread PUBLIC ax_core_worker)


target_include_directories(ax_core_worker
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)