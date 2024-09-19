add_library(ax_core_math STATIC
        math_acos_lt.cpp
        math_random.cpp
)


target_link_libraries(ax_core_common PUBLIC ax_core_math)
target_link_libraries(ax_core_math
PRIVATE
        ax_core_macro
        Boost::random
        Boost::math
PUBLIC
        Boost::container_hash
        ax_core_device
        glm::glm

)


target_include_directories(ax_core_math
PUBLIC
    ${AXOMAE_CORELIB_ROOT}
)

set(PRECISION_LUT "")
if(AXOMAE_LUT_HIGHP)
    set(PRECISION_LUT 1000000)
elseif (AXOMAE_LUT_MEDIUMP)
    set(PRECISION_LUT 500000)
else()
  set(PRECISION_LUT 10000)
endif()

target_compile_definitions(ax_core_math PRIVATE PRECISION_LUT=${PRECISION_LUT})

if(AXOMAE_USE_CUDA)
  autoload_build_utils()
  target_compile_definitions(ax_core_math PUBLIC AXOMAE_USE_CUDA)
  target_sources(ax_core_math PRIVATE gpu/math_random_gpu.cpp)
  target_link_libraries(ax_core_math PRIVATE
        CUDA::curand
  )
  register_device_compilation(gpu/math_random_gpu.cpp)

endif()



set(GEN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/lib/lookup_tables/binary/")



file(MAKE_DIRECTORY ${GEN_INCLUDE_DIR})
set(ACOS_TABLE_NAME "acos_table")
set(ACOS_OUTPUT ${GEN_INCLUDE_DIR}/${ACOS_TABLE_NAME}.o)

add_custom_command(
  OUTPUT ${ACOS_OUTPUT} 
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_luts_trigonometrics.py ${PRECISION_LUT} ${ACOS_TABLE_NAME}
  DEPENDS ${PROJECT_SOURCE_DIR}/scripts/generate_luts_trigonometrics.py
  WORKING_DIRECTORY ${GEN_INCLUDE_DIR}
  COMMENT "Generating trigonometric lookup tables."
  VERBATIM
)

add_custom_target(axcustom_lut_generate 
  DEPENDS ${ACOS_OUTPUT}
)

set(SOBOL_DV_NAME "sobol_direction_vectors")
set(SOBOL_OUTPUT ${GEN_INCLUDE_DIR}/${SOBOL_DV_NAME}.h)
set(JOE_KUO_DATASET ${PROJECT_SOURCE_DIR}/scripts/joe_kuo.txt)
set(DIMENSIONS_SIZE 256)
set(SEQUENCE_SIZE 32)

target_compile_definitions(ax_core_math PUBLIC DIMENSIONS_SIZE=${DIMENSIONS_SIZE})
target_compile_definitions(ax_core_math PUBLIC SEQUENCE_SIZE=${SEQUENCE_SIZE})
add_custom_command(
  OUTPUT ${SOBOL_OUTPUT}
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_luts_sobol.py -i ${JOE_KUO_DATASET} -d ${DIMENSIONS_SIZE} -s ${SEQUENCE_SIZE} -o ${SOBOL_DV_NAME}
  DEPENDS ${PROJECT_SOURCE_DIR}/scripts/generate_luts_sobol.py 
  WORKING_DIRECTORY ${GEN_INCLUDE_DIR}
  COMMENT "Generating Sobol direction vectors."
  VERBATIM
)

add_custom_target(axcustom_sobol_generate DEPENDS ${SOBOL_OUTPUT})

target_link_libraries(ax_core_math PRIVATE
  ${ACOS_OUTPUT} 
)
target_include_directories(ax_core_math PUBLIC 
  ${GEN_INCLUDE_DIR}
)

add_dependencies(ax_core_math axcustom_lut_generate axcustom_sobol_generate)


