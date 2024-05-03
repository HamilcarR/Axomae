add_library(ax_math
        math_acos_lt.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC ax_math)
target_link_libraries(ax_math
PUBLIC
        ax_macro
)


target_include_directories(ax_math 
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vector
)

if(HIGHP_LUT)
    target_compile_definitions(ax_math PUBLIC HIGHP_LUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-loop-limit=1000000")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-ops-limit=600000000")
elseif (MEDIUMP_LUT)
    target_compile_definitions(ax_math PUBLIC MEDIUMP_LUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-loop-limit=500000")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-ops-limit=60000000")
else()
    target_compile_definitions(ax_math PUBLIC LOWP_LUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-loop-limit=10000")
endif()
