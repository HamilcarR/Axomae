add_library(ax_core_math
        math_acos_lt.cpp
)
target_link_libraries(ax_core_common PUBLIC ax_core_math)
target_link_libraries(ax_core_math
PRIVATE
        ax_core_macro
PUBLIC
        Boost::random
)


target_include_directories(ax_core_math
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vector
    ${GLM_INCLUDE_PATH}
)



function(precision_compile_flag precision)
    set(FLAGS "${CMAKE_CXX_FLAGS}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if(precision MATCHES "LOW")
            set(FLAGS "${FLAGS} -fconstexpr-loop-limit=10000")
        elseif (precision MATCHES "MED")
            set(FLAGS "${FLAGS} -fconstexpr-loop-limit=500000")
            set(FLAGS "${FLAGS} -fconstexpr-ops-limit=60000000")
        else()
            set(FLAGS "${FLAGS} -fconstexpr-loop-limit=1000000")
            set(FLAGS "${FLAGS} -fconstexpr-ops-limit=600000000")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(precision MATCHES "LOW")
            set(FLAGS "${FLAGS} -fconstexpr-steps=10000")
        elseif (precision MATCHES "MED")
            set(FLAGS "${FLAGS} -fconstexpr-steps=500000")
        else()
            set(FLAGS "${FLAGS} -fconstexpr-steps=600000000")
        endif()
    endif()
    set(CMAKE_CXX_FLAGS "${FLAGS}" PARENT_SCOPE)
endfunction()


if(AXOMAE_LUT_HIGHP)
    target_compile_definitions(ax_core_math PUBLIC AXOMAE_LUT_HIGHP)
    precision_compile_flag(HIGH)
elseif (AXOMAE_LUT_MEDIUMP)
    target_compile_definitions(ax_core_math PUBLIC AXOMAE_LUT_MEDIUMP)
    precision_compile_flag(MED)
else()
    target_compile_definitions(ax_core_math PUBLIC AXOMAE_LUT_LOWP)
    precision_compile_flag(LOW)
endif()
