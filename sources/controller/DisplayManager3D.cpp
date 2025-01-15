#include "DisplayManager3D.h"

#include "Config.h"
#include "RenderPipeline.h"
#include "nova/bake.h"

namespace exception {
  class SceneTreeInitializationException : public CatastrophicFailureException {
   public:
    static constexpr const char *text = "Realtime viewer not initialized, cannot retrieve scene tree.";
    SceneTreeInitializationException() { saveErrorString(text); }
  };
}  // namespace exception

namespace controller {

  void DisplayManager3D::connect_slots() {
    connect(this, &DisplayManager3D::signal_halt_renderers, realtime_viewer, &GLViewer::haltRender);
    connect(this, &DisplayManager3D::signal_resume_renderers, realtime_viewer, &GLViewer::resumeRender);
    connect(this, &DisplayManager3D::signal_sync_renderers, realtime_viewer, &GLViewer::syncRenderer);
    connect(this, &DisplayManager3D::signal_switch_realtime_ctx, realtime_viewer, &GLViewer::currentCtx);
    connect(this, &DisplayManager3D::signal_done_realtime_ctx, realtime_viewer, &GLViewer::doneCtx);
  }

  void DisplayManager3D::init(Ui::MainWindow &main_window_ui,
                              ApplicationConfig *global_application_config,
                              OperatorProgressStatus *progress_manager) {

    /* Realtime renderer initialization*/
    realtime_viewer = main_window_ui.renderer_view;

    realtime_viewer->getRenderer().getRenderPipeline().setProgressManager(progress_manager);
    realtime_viewer->setProgressManager(progress_manager);
    realtime_viewer->setApplicationConfig(global_application_config);

    /* Nova raytracer */
    nova_viewer = main_window_ui.nova_viewer->getViewer();
    nova_viewer->renderOnTimer(0);
    nova_viewer->getRenderer().getRenderPipeline().setProgressManager(progress_manager);
    nova_viewer->setProgressManager(progress_manager);
    nova_viewer->setApplicationConfig(global_application_config);

    nova_resource_manager = std::make_unique<nova::NovaResourceManager>();
    connect_slots();
  }
  void DisplayManager3D::makeCtxRealtime() { emit signal_switch_realtime_ctx(); }
  void DisplayManager3D::doneCtxRealtime() { emit signal_done_realtime_ctx(); }
  void DisplayManager3D::haltRenderers() { emit signal_halt_renderers(); }
  void DisplayManager3D::resumeRenderers() { emit signal_resume_renderers(); }

  template<class T>
  axstd::span<uint8_t> convert_to_byte_span(const axstd::span<T> &to_convert) {
    std::size_t size_in_bytes = sizeof(T) * to_convert.size();
    auto buffer_in_bytes = reinterpret_cast<uint8_t *>(to_convert.data());
    return {buffer_in_bytes, size_in_bytes};
  }

  static void add_caches_addresses(nova::device_shared_caches_t &shared_caches,
                                   const nova_baker_utils::bake_buffers_storage_t &bake_buffers_storage) {
    /* Register textures */
    shared_caches.addSharedCacheAddress(convert_to_byte_span(bake_buffers_storage.texture_buffers.image_alloc_buffer));
    /* Register materials */
    shared_caches.addSharedCacheAddress(convert_to_byte_span(bake_buffers_storage.material_buffers.diffuse_alloc_buffer));
    shared_caches.addSharedCacheAddress(convert_to_byte_span(bake_buffers_storage.material_buffers.conductor_alloc_buffer));
    shared_caches.addSharedCacheAddress(convert_to_byte_span(bake_buffers_storage.material_buffers.dielectric_alloc_buffer));
    /* Register shapes */
    shared_caches.addSharedCacheAddress(convert_to_byte_span(bake_buffers_storage.primitive_buffers.triangle_alloc_buffer));
    /*Register primitives */
    shared_caches.addSharedCacheAddress(convert_to_byte_span(bake_buffers_storage.primitive_buffers.geo_primitive_alloc_buffer));
  }

  static void notify_simple_progress(const char *str, IProgressManager &progress_manager) {
    progress_manager.initProgress(str, 100.f);
    progress_manager.setCurrent(90);
    progress_manager.notifyProgress();
  }

  struct mesh_transform_t {
    Mesh *mesh;
    glm::mat4 original_transfo;
  };

  static std::vector<mesh_transform_t> retrieve_original_mesh_transform(const SceneChangeData &scene_change) {
    std::vector<mesh_transform_t> mesh_transf;
    for (Mesh *elem : scene_change.mesh_list) {
      mesh_transf.push_back({elem, elem->computeFinalTransformation()});
    }
    return mesh_transf;
  }

  /* Allows to retrieve only the meshes of the imported scene file ... Other meshes like skyboxes , bounding boxes , sprites are discarded . */
  static std::vector<nova_baker_utils::drawable_original_transform> retrieve_scene_main_drawables(
      const std::vector<Drawable *> &scene_retrieved_drawables_ptrs, const std::vector<mesh_transform_t> &loaded_meshes) {

    std::vector<nova_baker_utils::drawable_original_transform> scene_mesh_drawables;
    scene_mesh_drawables.reserve(loaded_meshes.size());

    for (Drawable *drawable : scene_retrieved_drawables_ptrs)
      for (const mesh_transform_t &mesh_trf : loaded_meshes)
        if (drawable->getMeshPointer() == mesh_trf.mesh)
          scene_mesh_drawables.push_back({drawable, mesh_trf.original_transfo});
    return scene_mesh_drawables;
  }

  void DisplayManager3D::setNewScene(SceneChangeData &scene_data, ProgressStatus *progress_status) {
    prepareSceneChange();
    scene_data.nova_resource_manager = nova_resource_manager.get();
    IProgressManager progress_manager;
    progress_manager.setProgressManager(progress_status);
    nova_resource_manager->clearResources();
    std::vector<mesh_transform_t> original_transforms = retrieve_original_mesh_transform(scene_data);
    notify_simple_progress("Initializing Nova , building BVH and allocating shared GPU caches", progress_manager);

    /* Scene initialization modifies the tree. */
    realtime_viewer->setNewScene(scene_data);
    nova_viewer->setNewScene(scene_data);

    const RendererInterface &realtime_renderer = realtime_viewer->getRenderer();
    const Scene &realtime_scene = realtime_renderer.getScene();
    auto drawable_collection = retrieve_scene_main_drawables(realtime_scene.getDrawables(), original_transforms);
    realtime_viewer->makeCurrent();
    bake_buffers_storage = nova_baker_utils::build_scene(drawable_collection, *nova_resource_manager);
    realtime_viewer->doneCurrent();
    add_caches_addresses(shared_caches, bake_buffers_storage);
    /* Build acceleration. */
    nova::aggregate::Accelerator accelerator = nova_baker_utils::build_performance_acceleration_structure(
        nova_resource_manager->getPrimitiveData().get_primitives());
    nova_resource_manager->setAccelerationStructure(accelerator);

    progress_manager.reset();
  }

  void DisplayManager3D::prepareSceneChange() {
    realtime_viewer->prepareRendererSceneChange();
    nova_viewer->prepareRendererSceneChange();
  }

  SceneTree &DisplayManager3D::getSceneTree() const {
    if (!realtime_viewer) {
      throw exception::SceneTreeInitializationException();
    }
    return realtime_viewer->getRenderer().getScene().getSceneTreeRef();
  }

}  // namespace controller