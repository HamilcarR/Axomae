#include "DisplayManager3D.h"
#include "Config.h"
#include "RenderPipeline.h"
#include "api_common.h"
#include "internal/macro/project_macros.h"
#include "nova/bake.h"
#include <nova/api_engine.h>

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

    // Nova engine creation
    nova_engine = nova::create_engine();
    nova::ERROR_STATE err = nova::init_threads(global_application_config->getThreadPoolSize());
    AX_ASSERT_EQ(err, nova::SUCCESS);

    // Setup options
    nova::RenderOptions *render_options = nova_engine->getRenderOptions();
    AX_ASSERT_NOTNULL(render_options);
    render_options->useInterops(true);

    // Setup render buffers
    nova::RenderBuffer *render_buffers = nova_engine->getRenderBuffers();
    AX_ASSERT_NOTNULL(render_buffers);
    unsigned INITIAL_WIDTH = 500;
    unsigned INITIAL_HEIGHT = 500;
    err = render_buffers->preallocate(INITIAL_WIDTH, INITIAL_HEIGHT);
    AX_ASSERT_EQ(err, nova::SUCCESS);

    SceneChangeData scene_data;
    scene_data.nova_engine = nova_engine.get();
    nova_viewer->setNewScene(scene_data);
    connect_slots();
  }
  void DisplayManager3D::makeCtxRealtime() { emit signal_switch_realtime_ctx(); }

  void DisplayManager3D::doneCtxRealtime() { emit signal_done_realtime_ctx(); }

  void DisplayManager3D::haltRenderers() { emit signal_halt_renderers(); }

  void DisplayManager3D::resumeRenderers() { emit signal_resume_renderers(); }

  void DisplayManager3D::onEnvmapChange() { nova_viewer->signalEnvmapChange(); }

  static void notify_simple_progress(const char *str, IProgressManager &progress_manager) {
    progress_manager.initProgress(str, 100.f);
    progress_manager.setCurrent(90);
    progress_manager.notifyProgress();
  }

  struct mesh_transform_t {
    Mesh *mesh;
    glm::mat4 original_transfo;
  };

  /* Rotates the entire scene 90° counter clock wise on the X axis.
   * This is done because Nova does spherical envmapping and doesn't transform sampling vectors.
   * IE, it just syncs the orientation of the scene in the realtime viewer, and the orientation of the scene in Nova.
   * There's also a correction of the view matrix in nova_baker_utils::initialize_scene_data().
   */
  static glm::mat4 transform_swizzle(const Mesh *elem) {
    return glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f)) * elem->getAccumulatedModelMatrix() *
           elem->getLocalModelMatrix();
  }

  static std::vector<mesh_transform_t> retrieve_original_mesh_transform(const SceneChangeData &scene_change) {
    std::vector<mesh_transform_t> mesh_transf;
    for (Mesh *elem : scene_change.mesh_list) {
      mesh_transf.push_back({elem, transform_swizzle(elem)});
    }
    return mesh_transf;
  }

  /* Retrieves only the geometric meshes of the imported scene file ... Other meshes like skyboxes , bounding boxes , sprites are discarded . */
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
    scene_data.nova_engine = nova_engine.get();
    IProgressManager progress_manager;
    progress_manager.setProgressManager(progress_status);

    std::vector<mesh_transform_t> original_transforms = retrieve_original_mesh_transform(scene_data);
    notify_simple_progress("Initializing Nova , building BVH and allocating shared GPU caches", progress_manager);

    /* Scene initialization modifies the tree. */
    realtime_viewer->setNewScene(scene_data);
    nova_viewer->setNewScene(scene_data);

    const RendererInterface &realtime_renderer = realtime_viewer->getRenderer();
    const Scene &realtime_scene = realtime_renderer.getScene();
    auto drawable_collection = retrieve_scene_main_drawables(realtime_scene.getDrawables(), original_transforms);
    realtime_viewer->makeCurrent();
    nova::Scene *nova_scene = nova_engine->getScene();
    AX_ASSERT_NOTNULL(nova_scene);
    nova_baker_utils::build_scene(drawable_collection, nova_scene);
    nova_engine->buildScene();
    realtime_viewer->doneCurrent();
    nova_engine->buildAcceleration();
    progress_manager.reset();
  }

  void DisplayManager3D::prepareSceneChange() {
    nova_engine->cleanup();
    nova_viewer->signalEnvmapChange();
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
