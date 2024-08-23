function(set_fpic target)
    message(STATUS "Setting -FPIC for target: "${target})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()



