 target_sources(ax_core_nova_aggregate PRIVATE
          host.cpp
          device.cpp
          programs.cu
        )
  target_include_directories(ax_core_nova_aggregate PRIVATE ${OPTIX_INCLUDE_PATH})
 
  target_compile_definitions(ax_core_nova_aggregate PUBLIC AXOMAE_USE_OPTIX)

  register_device_compilation(device.cpp)


