target_sources(ax_core_nova PRIVATE
        native/BvhBuilder.cpp
        native/native.cpp
        api/device_accelerator_factory.cpp
)

if(AXOMAE_USE_EMBREE)
  target_sources(ax_core_nova PRIVATE api/embree/embree.cpp) 
  target_link_libraries(ax_core_nova PRIVATE embree)
endif(AXOMAE_USE_EMBREE)

if(AXOMAE_USE_CUDA)
  autoload_build_utils()
  if(AXOMAE_USE_OPTIX)
    add_subdirectory(api/optix)
  endif()
endif()

