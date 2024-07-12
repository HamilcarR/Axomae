add_library(ax_core_thread
    Mutex.cpp
)

target_link_libraries(ax_core PUBLIC ax_core_thread)

target_include_directories(ax_core_thread
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

)

add_subdirectory(worker)