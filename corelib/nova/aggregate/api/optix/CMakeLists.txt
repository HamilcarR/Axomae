# We compile optix programs to ptx.
add_library(ax_ptx_compile OBJECT 
  ${CUDA_OPTIX_PROGRAMS}
)
set_target_properties(ax_ptx_compile PROPERTIES
        CUDA_PTX_COMPILATION ON)

 target_include_directories(ax_ptx_compile PUBLIC 
    ${AXOMAE_NOVA_ROOT}
    ${AXOMAE_CORELIB_ROOT}
    ${OPTIX_INCLUDE_PATH}
    ${CMAKE_BINARY_DIR}/lib/lookup_tables/binary/
  )

  target_link_libraries(ax_ptx_compile PRIVATE
          ax_core_geometry
          ax_core_math
          ax_core_macro
          ax_core_debug
          ax_core_thread
          ax_core_memory
          ax_core_device
          ax_core_nova_sampler
          ax_core_nova_texturing
          ax_core_nova_primitive
          ax_core_nova_shape
          ax_core_nova_material
  )

set(OUTPUT_PTX ${CMAKE_BINARY_DIR}/lib/ptx/integrator_optix)
set(EMBED_PTX_VAR_NAME "PTX_EMBEDDED")
set(AX_PTX_INTEGRATOR ${OUTPUT_PTX}.ptx_embedded.c )
add_custom_command(
  OUTPUT ${AX_PTX_INTEGRATOR}
  COMMAND ${CMAKE_COMMAND}
  "-DBIN_TO_C_COMMAND=${BIN2C}"
  "-DOBJECTS=$<TARGET_OBJECTS:ax_ptx_compile>"
  "-DVAR_NAME=${EMBED_PTX_VAR_NAME}"
  "-DOUTPUT=${AX_PTX_INTEGRATOR}"
  -P "${AXOMAE_CORELIB_ROOT}/cmake/bin2c_wrapper.cmake"
  VERBATIM
  DEPENDS ${CUDA_OPTIX_PROGRAMS} $<TARGET_OBJECTS:ax_ptx_compile> 
  COMMENT "Generating GPU integrators PTX files."
)

add_custom_target(ax_ptx_generate DEPENDS ${AX_PTX_INTEGRATOR})
add_dependencies(ax_ptx_generate ax_ptx_compile) 



add_library(ax_ptx_embedded_lib STATIC ${AX_PTX_INTEGRATOR})
add_dependencies(ax_ptx_embedded_lib ax_ptx_generate)        
target_link_libraries(ax_core_nova_aggregate PRIVATE ax_ptx_embedded_lib)

target_sources(ax_core_nova_aggregate PRIVATE
          host.cpp
          device.cpp
)

target_include_directories(ax_core_nova_aggregate PRIVATE ${OPTIX_INCLUDE_PATH}) 
target_compile_definitions(ax_core_nova_aggregate PUBLIC AXOMAE_USE_OPTIX)


