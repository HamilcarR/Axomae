#include "aggregate/acceleration_interface.h"
#include "aggregate/aggregate_datastructures.h"
#include "api_common.h"
#include "api_renderoptions.h"
#include "engine/nova_engine.h"
#include "internal/thread/worker/ThreadPool.h"
#include "manager/NovaExceptionManager.h"
#include "manager/NovaResourceManager.h"
#include "primitive/PrimitiveInterface.h"
#include "private_includes.h"
#include "shape/MeshContext.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/nova_texturing.h"
#include <internal/common/utils.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>
#include <memory>

using RenderEngineInterfacePtr = std::unique_ptr<NovaRenderEngineInterface>;
using EngineExceptionManagerPtr = std::unique_ptr<nova::NovaExceptionManager>;
using ThreadpoolPtr = std::unique_ptr<threading::ThreadPool>;

static const char *const _NOVA_HOST_TAG = "_NOVA_HOST_TAG";
namespace nova {
  class NvEngineInstance : public Engine {
    NovaResourceManager manager;
    ThreadpoolPtr threadpool;
    EngineExceptionManagerPtr engine_exception_manager;
    RenderBufferPtr render_buffer;
    RenderOptionsPtr render_options;  // is synced with the manager's data.
    RenderEngineInterfacePtr render_engine;
    ScenePtr scene;
    bool scene_built{false};

   public:
    NvEngineInstance() {
      render_options = create_renderoptions();
      render_buffer = create_renderbuffer();
      scene = create_scene();

      engine_exception_manager = std::make_unique<NovaExceptionManager>();
      render_engine = std::make_unique<NovaRenderEngineLR>();
    }

    ERROR_STATE setThreadSize(unsigned threads) override {
      try {
        threadpool = std::make_unique<threading::ThreadPool>(threads);
      } catch (...) {
        return THREADPOOL_CREATION_ERROR;
      }
      return SUCCESS;
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

    static shape::triangle::mesh_vbo_ids create_vbo_pack(const Trimesh &mesh) {
      shape::triangle::mesh_vbo_ids vbo_pack{};
      vbo_pack.vbo_positions = mesh.getInteropVertices();
      vbo_pack.vbo_indices = mesh.getInteropIndices();
      vbo_pack.vbo_normals = mesh.getInteropNormals();
      vbo_pack.vbo_uv = mesh.getInteropUVs();
      vbo_pack.vbo_tangents = mesh.getInteropTangents();
      return vbo_pack;
    }

    struct triangle_mesh_properties_t {
      std::size_t mesh_index;
      std::size_t triangle_index;
    };

    static glm::mat4 convert_transform(const float transform[16]) {
      glm::mat4 final_transform;
      for (int i = 0; i < 16; i++)
        glm::value_ptr(final_transform)[i] = transform[i];
      return final_transform;
    }

    static void add_trimesh_geometry(const Trimesh &mesh, const glm::mat4 &final_transform, NovaResourceManager &manager, bool use_interops) {
      nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
      const Object3D &geometry = to_obj3d(mesh);
      std::size_t stored_mesh_index = res_holder.addTriangleMesh(geometry);

      res_holder.addTransform(final_transform, stored_mesh_index);
      if (core::build::is_gpu_build && use_interops) {
        shape::triangle::mesh_vbo_ids vbo_pack = create_vbo_pack(mesh);
        res_holder.addTriangleMesh(vbo_pack);
      }
    }

    static void setup_trimesh(const Trimesh &mesh, const Material &material, NovaResourceManager &manager, bool uses_interops, size_t mesh_index) {
      const Object3D &geometry = to_obj3d(mesh);

      float transform_array[16] = {};
      const Transform &transform = mesh.getTransform();
      transform.getTransformMatrix(transform_array);
      const glm::mat4 transform_matrix = convert_transform(transform_array);

      add_trimesh_geometry(mesh, transform_matrix, manager, uses_interops);

      material::NovaMaterialInterface material_interface = setup_material_data(mesh, material, manager);
      for (std::size_t triangle_index = 0; triangle_index < geometry.indices.size(); triangle_index += Object3D::face_stride) {
        nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
        auto triangle = res_holder.addShape<nova::shape::Triangle>(mesh_index, triangle_index);
        manager.getPrimitiveData().addPrimitive<nova::primitive::NovaGeoPrimitive>(triangle, material_interface);
      }
    }

    static void setup_triangle_meshes(const Scene &scene, NovaResourceManager &manager, bool uses_interops) {
      std::size_t mesh_index = 0;
      for (size_t i = 0; i < scene.getMeshesNum(mesh::TRIANGLE); i++) {
        const Trimesh &geometry = *scene.getTriangleMeshCollection()[i];
        const Material &material = *scene.getMaterialCollection(mesh::TRIANGLE)[i];
        setup_trimesh(geometry, material, manager, uses_interops, mesh_index++);
      }
    }

    static void setup_meshes(const Scene &scene, NovaResourceManager &manager, bool uses_interops) {
      setup_triangle_meshes(scene, manager, uses_interops);
    }

    ERROR_STATE setScene(ScenePtr ptr) override {
      if (!ptr)
        return INVALID_ARGUMENT;
      scene = std::move(ptr);
      return SUCCESS;
    }

    const Scene *getScene() const override { return scene.get(); }

    Scene *getScene() override { return scene.get(); }

    ERROR_STATE buildScene() override {
      std::size_t number_primitives = scene->getPrimitivesNum(mesh::TRIANGLE);
      std::size_t number_trimesh = scene->getMeshesNum(mesh::TRIANGLE);
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
      setup_meshes(*scene, manager, render_options->isUsingInterops());

      scene_built = true;
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

    ERROR_STATE buildAcceleration() override {
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

    ERROR_STATE cleanup() override {
      manager.clearResources();
      scene_built = false;
      if (!render_buffer)
        return INVALID_BUFFER_STATE;
      render_buffer->resetBuffers();
      return SUCCESS;
    }

    ERROR_STATE setRenderBuffers(RenderBufferPtr buffers) override {
      if (!buffers)
        return INVALID_BUFFER_STATE;
      render_buffer = std::move(buffers);
      return SUCCESS;
    }

    ERROR_STATE stopRender() override {
      manager.getEngineData().is_rendering = false;
      if (!render_options->isUsingGpu()) {
        if (!threadpool)
          return THREADPOOL_NOT_INITIALIZED;
        /* Empty scheduler list. */
        threadpool->emptyQueue(_NOVA_HOST_TAG);
        /* Synchronize the threads. */
        threadpool->fence(_NOVA_HOST_TAG);
      }
      /* Cleanup gpu resources */
      manager.getShapeData().releaseResources();
      manager.getTexturesData().releaseResources();

      return SUCCESS;
    }

    static void setup_engine_data(engine::EngineResourcesHolder &resrc, const RenderOptions &render_options) {
      resrc.aliasing_samples = render_options.getAliasingSamples();
      resrc.max_depth = render_options.getMaxDepth();
      resrc.sample_increment = render_options.getSamplesIncrement();
      resrc.vertical_invert = render_options.isFlippedV();
      resrc.tiles_width = render_options.getTileDimensionWidth();
      resrc.tiles_height = render_options.getTileDimensionHeight();
      resrc.integrator_flag = render_options.getIntegratorFlag();
    }

    static void allocate_environment_maps(texturing::TextureResourcesHolder &resrc, const nova::Scene &scene) {
      resrc.allocateEnvironmentMaps(scene.getEnvmapCollection().size());
      resrc.setEnvmapId(scene.getCurrentEnvmapId());
      for (const TexturePtr &texture : scene.getEnvmapCollection()) {
        AX_ASSERT_EQ(texture->getFormat(), texture::FLOATX4);
        std::size_t index = resrc.addTexture(
            static_cast<const float *>(texture->getTextureBuffer()), texture->getWidth(), texture->getHeight(), texture->getChannels());
        resrc.addNovaTexture<texturing::EnvmapTexture>(index);
      }
    }

    ERROR_STATE prepareRender() override {
      texturing::TextureResourcesHolder &textures_manager = manager.getTexturesData();
      shape::ShapeResourcesHolder &shape_manager = manager.getShapeData();
      engine::EngineResourcesHolder &opts_manager = manager.getEngineData();
      setup_engine_data(opts_manager, *render_options);
      allocate_environment_maps(textures_manager, *scene);
      if (render_options->isUsingGpu()) {
        shape_manager.lockResources();
        shape_manager.mapBuffers();
        textures_manager.lockResources();
        textures_manager.mapBuffers();
      }
      return SUCCESS;
    }

    ERROR_STATE startRender(HdrBufferStruct *buffers) override { return INVALID_ARGUMENT; }

    ERROR_STATE startRender() override {
      manager.getEngineData().is_rendering = true;
      return SUCCESS;
    }

    ERROR_STATE setRenderOptions(RenderOptionsPtr opts) override {
      if (!opts)
        return INVALID_ARGUMENT;
      render_options = std::move(opts);
      return SUCCESS;
    }

    const RenderOptions *getRenderOptions() const override { return render_options.get(); }

    RenderOptions *getRenderOptions() override { return render_options.get(); }

    const RenderBuffer *getRenderBuffers() const override { return render_buffer.get(); }

    RenderBuffer *getRenderBuffers() override { return render_buffer.get(); }

    ERROR_STATE synchronize() override {
      if (!threadpool)
        return THREADPOOL_NOT_INITIALIZED;
      threadpool->fence(_NOVA_HOST_TAG);
      return SUCCESS;
    }

    NovaResourceManager &getResrcManager() override { return manager; }
  };

  EnginePtr create_engine() { return std::make_unique<NvEngineInstance>(); }

}  // namespace nova
