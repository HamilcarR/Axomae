
set(AXOMAE_CORELIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
function(autoload_build_utils)
    get_property(utils_loaded GLOBAL PROPERTY UTILS_LOADED)
    if(NOT utils_loaded)
        include(${AXOMAE_CORELIB_ROOT}/cmake/utils.cmake)
        set_property(GLOBAL PROPERTY UTILS_LOADED TRUE)
    endif ()
endfunction()

add_library(ax_core STATIC)

target_include_directories(ax_core
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


add_subdirectory(internal)
add_subdirectory(nova)

