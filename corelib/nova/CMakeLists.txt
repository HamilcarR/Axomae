set(AXOMAE_NOVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_library(ax_core_nova STATIC
        DrawEngine.cpp
        DrawEngine_GPU.cpp
        nova_gpu_utils.cpp
        
        texturing/nova_texturing.cpp 
        texturing/texture_interop_storage.cpp

        shape/Sphere.cpp
        shape/Square.cpp
        shape/Box.cpp
        shape/nova_shape.cpp
        shape/triangle_mesh_storage.cpp
        shape/mesh_transform_storage.cpp

        scene/nova_scene.cpp

        engine/nova_engine.cpp 
        engine/nova_exception.cpp

        primitive/nova_primitive.cpp

        manager/NovaExceptionManager.cpp
        manager/NovaResourceManager.cpp

        integrator/Integrator.cpp
        integrator/DepthIntegrator.cpp
        integrator/NormalIntegrator.cpp
        integrator/EmissiveIntegrator.cpp
        integrator/EmissiveIntegrator.h

)

target_link_libraries(ax_core PUBLIC ax_core_nova)

target_link_libraries(ax_core_nova
PRIVATE
        ax_core_common
        ax_core_geometry
        ax_core_memory
        ax_core_thread
        ax_core_macro
        ax_core_device
        ax_core_debug
)

target_include_directories(ax_core_nova
  PUBLIC
        ${AXOMAE_CORELIB_ROOT}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(AXOMAE_USE_CUDA)
    set_source_files_properties(DrawEngine_GPU.cpp PROPERTIES LANGUAGE CUDA) #TODO : probably useless
    target_compile_definitions(ax_core_nova PRIVATE AXOMAE_USE_CUDA)
    if(AXOMAE_USE_OPTIX)
        target_compile_definitions(ax_core_nova PRIVATE AXOMAE_USE_OPTIX)
        target_include_directories(ax_core_nova PRIVATE ${OPTIX_INCLUDE_PATH})
        set(CUDA_OPTIX_PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/integrator/cuda/integrator_optix.cu)  
    endif()
    target_sources(ax_core_nova PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/integrator/cuda/job_dispatch.cpp)
endif()

add_subdirectory(aggregate)

