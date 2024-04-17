add_library(ax_worker
)

target_link_libraries(ax_thread PUBLIC ax_worker)


target_include_directories(ax_worker
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)