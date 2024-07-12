add_library(ax_core_exception
    GenericException.cpp 
) 

target_link_libraries(ax_core_common PUBLIC ax_core_exception)
target_link_libraries(ax_core_exception
 PRIVATE
        ax_core_macro
)

target_include_directories(ax_core_exception
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

)