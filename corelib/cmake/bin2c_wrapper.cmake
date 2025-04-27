# https://github.com/robertmaynard/code-samples/blob/master/posts/cmake_ptx/bin2c_wrapper.cmake
set(file_contents)
foreach(obj ${OBJECTS})
  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_name ${obj} NAME_WE)
  get_filename_component(obj_dir ${obj} DIRECTORY)
  if(obj_ext MATCHES ".ptx" AND NOT obj_name MATCHES "mocs_*")
    set(args --length --name ${VAR_NAME} ${obj})
    execute_process(COMMAND "${BIN_TO_C_COMMAND}" ${args}
                    WORKING_DIRECTORY ${obj_dir}
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE output
                    ERROR_VARIABLE error_var
                    )

    set(file_contents "#include<stdint.h> \n${file_contents} \n${output}")
  endif()
endforeach()
file(WRITE "${OUTPUT}" "${file_contents}")
