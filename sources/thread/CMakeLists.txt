add_library(ax_thread 
    Mutex.cpp
        ThreadPool.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC ax_thread)

target_include_directories(ax_thread 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)