#include "aggregate/acceleration_interface.h"
#include "aggregate/aggregate_datastructures.h"
#include "api_common.h"
#include "engine/datastructures.h"
#include "internal/common/utils.h"
#include "primitive/PrimitiveInterface.h"
#include "private_includes.h"
#include "shape/MeshContext.h"
#include <internal/macro/project_macros.h>
#include <memory>
#include <typeinfo>

namespace nova {

  ERROR_STATE NvRenderBuffer::createRenderBuffer(unsigned width, unsigned height) {
    render_buffers = std::make_unique<HostStoredRenderableBuffers>();
    render_buffers->width = width;
    render_buffers->height = height;
    try {
      render_buffers->partial_buffer.resize(width * height * HostStoredRenderableBuffers::CHANNELS_COLOR);
      render_buffers->accumulator_buffer.resize(width * height * HostStoredRenderableBuffers::CHANNELS_COLOR);
      render_buffers->depth_buffer.resize(width * height * HostStoredRenderableBuffers::CHANNELS_DEPTH);
      render_buffers->normal_buffer.resize(width * height * HostStoredRenderableBuffers::CHANNELS_NORMALS);
    } catch (...) {
      return OUT_OF_MEMORY;
    }
    return SUCCESS;
  }

  HdrBufferStruct NvRenderBuffer::getRenderBuffers() const {
    HdrBufferStruct buffers;
    buffers.accumulator_buffer = render_buffers->accumulator_buffer.data();
    buffers.partial_buffer = render_buffers->partial_buffer.data();
    buffers.normal_buffer = render_buffers->normal_buffer.data();
    buffers.depth_buffer = render_buffers->depth_buffer.data();
    buffers.channels = render_buffers->CHANNELS_COLOR;
    buffers.byte_size_color_buffers = render_buffers->width * render_buffers->height * render_buffers->CHANNELS_COLOR * sizeof(float);
    buffers.color_buffers_pitch = render_buffers->width * render_buffers->CHANNELS_COLOR * sizeof(float);
    buffers.depth_buffers_pitch = render_buffers->width * render_buffers->CHANNELS_DEPTH * sizeof(float);
    buffers.normal_buffers_pitch = render_buffers->width * render_buffers->CHANNELS_NORMALS * sizeof(float);
    return buffers;
  }

  struct resource_holders_inits_s {
    std::size_t triangle_mesh_number;
    std::size_t triangle_number;

    std::size_t dielectrics_number;
    std::size_t diffuse_number;
    std::size_t conductors_number;

    std::size_t image_texture_number;
    std::size_t constant_texture_number;
    std::size_t hdr_texture_number;
  };

  static void initialize_resources_holders(nova::NovaResourceManager &manager, const resource_holders_inits_s &resrc) {
    nova::shape::shape_init_record_s shape_init_data{};
    shape_init_data.total_triangle_meshes = resrc.triangle_mesh_number;
    shape_init_data.total_triangles = resrc.triangle_number;
    auto &shape_reshdr = manager.getShapeData();
    shape_reshdr.init(shape_init_data);

    nova::primitive::primitive_init_record_s primitive_init_data{};
    primitive_init_data.geometric_primitive_count = resrc.triangle_number;
    primitive_init_data.total_primitive_count = primitive_init_data.geometric_primitive_count;
    auto &primitive_reshdr = manager.getPrimitiveData();
    primitive_reshdr.init(primitive_init_data);

    nova::material::material_init_record_s material_init_data{};
    material_init_data.conductors_size = resrc.conductors_number;
    material_init_data.dielectrics_size = resrc.dielectrics_number;
    material_init_data.diffuse_size = resrc.diffuse_number;
    auto &material_resrc = manager.getMaterialData();
    material_resrc.init(material_init_data);

    nova::texturing::texture_init_record_s texture_init_data{};
    texture_init_data.total_constant_textures = resrc.constant_texture_number;
    texture_init_data.total_image_textures = resrc.image_texture_number;
    auto &texture_resrc = manager.getTexturesData();
    texture_resrc.allocateMeshTextures(texture_init_data);
  }

  struct triangle_mesh_properties_t {
    std::size_t mesh_index;
    std::size_t triangle_index;
  };
  static void store_primitive(const nova::material::NovaMaterialInterface &mat,
                              nova::NovaResourceManager &manager,
                              const triangle_mesh_properties_t &m_indices) {
    nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
    auto tri = res_holder.addShape<nova::shape::Triangle>(m_indices.mesh_index, m_indices.triangle_index);
    manager.getPrimitiveData().addPrimitive<nova::primitive::NovaGeoPrimitive>(tri, mat);
  }

  static std::size_t compute_primitive_number(const NvScene &scene) {
    std::size_t acc = 0;
    /*** Computes the number of triangle primitives in the whole scene. ***/
    auto trimesh_collection = scene.getTrimeshArray();
    const int trimesh_index_padding = 3;
    for (const auto &elem : trimesh_collection) {
      const Object3D &mesh_geometry = to_obj3d(*elem.mesh_geometry);
      AX_ASSERT_NOTNULL(elem.mesh_geometry);
      for (std::size_t i = 0; i < mesh_geometry.indices.size(); i += trimesh_index_padding)
        acc++;
    }

    /** Additional types of meshes here : */

    return acc;
  }

  static glm::mat4 convert_transform(const float transform[16]) {
    glm::mat4 final_transform;
    for (int i = 0; i < 16; i++) {
      glm::value_ptr(final_transform)[i] = transform[i];
    }
    return final_transform;
  }

  ERROR_STATE NvEngineInstance::buildScene(const NvAbstractScene &scene) {
    try {
      const NvScene &nv_scene = dynamic_cast<const NvScene &>(scene);
      std::size_t number_primitives = compute_primitive_number(nv_scene);
      std::size_t number_trimesh = nv_scene.getTrimeshArray().size();
      std::size_t total_mesh_number = number_trimesh;

      resource_holders_inits_s resrc{};
      resrc.triangle_mesh_number = total_mesh_number;
      resrc.triangle_number = number_primitives;
      resrc.conductors_number = total_mesh_number;
      resrc.diffuse_number = total_mesh_number;
      resrc.dielectrics_number = total_mesh_number;
      resrc.image_texture_number = total_mesh_number * PBR_TEXTURE_PACK_SIZE;
      resrc.hdr_texture_number = 1;  // At least 1 for Environment map.
      initialize_resources_holders(manager, resrc);
      std::size_t mesh_index = 0;
      for (const auto &elem : nv_scene.getTrimeshArray()) {
        const NvAbstractTriMesh &geometry = *elem.mesh_geometry.get();  // TODO: Other types of meshes for later
        const NvMaterial &material = *elem.mesh_material.get();
        material::NovaMaterialInterface material_interface = setup_material_data(geometry, material, manager);
        float transform[16]{};
        geometry.getTransform(transform);
        setup_geometry_data(geometry, transform, material_interface, manager, mesh_index++, uses_interops);
      }

    } catch (std::bad_cast &e) {
      return INVALID_SCENE_TYPE;
    }
    scene_built = true;
    return SUCCESS;
  }

  ERROR_STATE NvEngineInstance::useInterops(bool value) {
    if (!core::build::is_gpu_build) {
      uses_interops = false;
      return NOT_GPU_BUILD;
    }
    uses_interops = value;
    return SUCCESS;
  }

  ERROR_STATE NvEngineInstance::useGpu(bool gpu) {
    if (!core::build::is_gpu_build) {
      uses_gpu = false;
      return NOT_GPU_BUILD;
    }
    uses_gpu = gpu;
    return SUCCESS;
  }

  static aggregate::DefaultAccelerator build_cpu_managed_accelerator(const aggregate::primitive_aggregate_data_s &primitive_geometry) {
    aggregate::DefaultAccelerator accelerator;
    accelerator.build(primitive_geometry);
    return accelerator;
  }

  static std::unique_ptr<aggregate::DeviceAcceleratorInterface> build_device_accelerator(
      const aggregate::primitive_aggregate_data_s &primitive_geometry) {
    auto device_builder = aggregate::DeviceAcceleratorInterface::make();
    device_builder->build(primitive_geometry);
    return device_builder;
  }

  ERROR_STATE NvEngineInstance::buildAcceleration() {
    if (!scene_built)
      return SCENE_NOT_PROCESSED;
    primitive::primitives_view_tn primitive_list_view = manager.getPrimitiveData().getPrimitiveView();
    shape::MeshBundleViews mesh_geoemtry = manager.getShapeData().getMeshSharedViews();
    aggregate::primitive_aggregate_data_s aggregate;
    aggregate.primitive_list_view = primitive_list_view;
    aggregate.mesh_geometry = mesh_geoemtry;

    auto accelerator = build_cpu_managed_accelerator(aggregate);
    manager.setManagedCpuAccelerationStructure(std::move(accelerator));

    if (core::build::is_gpu_build) {
      auto device_accelerator = build_device_accelerator(aggregate);
      manager.setManagedGpuAccelerationStructure(std::move(device_accelerator));
    }

    return SUCCESS;
  }

  ERROR_STATE NvEngineInstance::cleanup() {
    manager.clearResources();
    uses_interops = false;
    uses_gpu = false;
    scene_built = false;

    return SUCCESS;
  }

  ERROR_STATE NvEngineInstance::setRenderBuffers(RenderBufferPtr buffers) {
    if (!buffers)
      return INVALID_BUFFER_STATE;
    render_buffer = std::move(buffers);
    return SUCCESS;
  }
}  // namespace nova
