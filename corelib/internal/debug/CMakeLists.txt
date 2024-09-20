add_library(ax_core_debug
    Logger.cpp
)

target_link_libraries(ax_core PUBLIC ax_core_debug)
target_link_libraries(ax_core_debug
PRIVATE
    ax_core_thread
    ax_core_macro
    ax_core_common
)
target_include_directories(ax_core_debug
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
 

)