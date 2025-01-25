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
set(ACOS_TABLE_NAME "acos_table")

file(MAKE_DIRECTORY ${GEN_INCLUDE_DIR})

add_custom_target(axcustom_lut_generate ALL
  COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_luts.py ${PRECISION_LUT} ${ACOS_TABLE_NAME}
  DEPENDS ${PROJECT_SOURCE_DIR}/scripts/generate_luts.py
  WORKING_DIRECTORY ${GEN_INCLUDE_DIR}
  COMMENT "Generating lookup tables."
  VERBATIM
)
target_link_libraries(ax_core_math PRIVATE
  ${GEN_INCLUDE_DIR}/${ACOS_TABLE_NAME}.o
)

add_dependencies(ax_core_math axcustom_lut_generate)


