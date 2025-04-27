add_library(ax_core_nova_aggregate STATIC
        native/BvhBuilder.cpp
        native/native.cpp
        api/device_accelerator_factory.cpp
)


target_link_libraries(ax_core_nova PUBLIC ax_core_nova_aggregate)

target_link_libraries(ax_core_nova_aggregate 
  PRIVATE
        ax_core_geometry
        ax_core_math
        ax_core_macro
        ax_core_nova_sampler
        ax_core_nova_primitive
        ax_core_memory
  PUBLIC
        ax_core_common
  )

if(AXOMAE_USE_EMBREE)
  target_sources(ax_core_nova_aggregate PRIVATE api/embree/embree.cpp) 
  target_link_libraries(ax_core_nova_aggregate PRIVATE embree )
endif(AXOMAE_USE_EMBREE)

if(AXOMAE_USE_CUDA)
  autoload_build_utils()
  if(AXOMAE_USE_OPTIX)
    add_subdirectory(api/optix)
  endif()

endif()


target_include_directories(ax_core_nova_aggregate PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}

)
