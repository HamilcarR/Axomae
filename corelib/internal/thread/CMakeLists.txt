add_library(ax_core_thread
    Mutex.cpp
)

target_link_libraries(ax_core PUBLIC ax_core_thread)

target_include_directories(ax_core_thread
PUBLIC 
    ${AXOMAE_CORELIB_ROOT}
)

add_subdirectory(worker)