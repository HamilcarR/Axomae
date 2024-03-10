add_library(ax_exception
    GenericException.cpp 
) 

target_link_libraries(${PROJECT_NAME} PUBLIC ax_exception)


target_include_directories(ax_exception 
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}

)