add_library(ax_core_nova_aggregate STATIC
        native/Bvh.cpp
        native/BvhBuilder.cpp
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
        ax_core_device
  PUBLIC 
        ax_core_common
  )

if(AXOMAE_USE_EMBREE)
  target_sources(ax_core_nova_aggregate PRIVATE api/embree.cpp) 
  target_link_libraries(ax_core_nova_aggregate PRIVATE embree )
endif(AXOMAE_USE_EMBREE)



target_include_directories(ax_core_nova_aggregate PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}      
)
