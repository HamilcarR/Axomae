add_library(ax_core_nova_integrator STATIC
        Integrator.cpp
        DepthIntegrator.cpp
        NormalIntegrator.cpp
        EmissiveIntegrator.cpp
        EmissiveIntegrator.h
)

target_link_libraries(ax_core_nova_integrator PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_debug
        ax_core_thread
        ax_core_nova_sampler
        ax_core_nova_texturing
        ax_core_nova_primitive
        ax_core_nova_aggregate
        ax_core_nova_manager
        ax_core_memory
        ax_core_device
)

target_link_libraries(ax_core_nova PUBLIC ax_core_nova_integrator)

target_include_directories(ax_core_nova_integrator
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

if(AXOMAE_USE_CUDA)
  if(AXOMAE_USE_OPTIX)
      set(CUDA_OPTIX_PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/integrator_optix.cu PARENT_SCOPE)  
  endif()
  target_sources(ax_core_nova_integrator PRIVATE 
          cuda/test.cu
          cuda/job_dispatch.cpp)
endif()
