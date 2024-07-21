
#include "GUIWindow.h"

#include "Loader.h"

namespace controller {

  void Controller::connect_all_slots() {

    /*Main Window -> Toolbar menu -> Files*/
    QObject::connect(main_window_ui.actionImport_image, SIGNAL(triggered()), this, SLOT(slot_import_image()));
    QObject::connect(main_window_ui.actionSave_image, SIGNAL(triggered()), this, SLOT(slot_save_image()));
    QObject::connect(main_window_ui.actionImport_Environment_Map, SIGNAL(triggered()), this, SLOT(slot_import_envmap()));

    QObject::connect(main_window_ui.use_average, SIGNAL(clicked()), this, SLOT(slot_greyscale_average()));
    QObject::connect(main_window_ui.use_luminance, SIGNAL(clicked()), this, SLOT(slot_greyscale_luminance()));
    QObject::connect(main_window_ui.use_scharr, SIGNAL(clicked()), this, SLOT(slot_use_scharr()));
    QObject::connect(main_window_ui.use_sobel, SIGNAL(clicked()), this, SLOT(slot_use_sobel()));
    QObject::connect(main_window_ui.use_prewitt, SIGNAL(clicked()), this, SLOT(slot_use_prewitt()));

    QObject::connect(main_window_ui.actionUndo, SIGNAL(triggered()), this, SLOT(slot_undo()));
    QObject::connect(main_window_ui.actionRedo, SIGNAL(triggered()), this, SLOT(slot_redo()));
    QObject::connect(main_window_ui.undo_button, SIGNAL(clicked()), this, SLOT(slot_undo()));
    QObject::connect(main_window_ui.redo_button, SIGNAL(clicked()), this, SLOT(slot_redo()));
    QObject::connect(main_window_ui.use_objectSpace, SIGNAL(clicked()), this, SLOT(slot_use_object_space()));
    QObject::connect(main_window_ui.use_tangentSpace, SIGNAL(clicked()), this, SLOT(slot_use_tangent_space()));

    QObject::connect(main_window_ui.smooth_dial, SIGNAL(valueChanged(int)), this, SLOT(slot_update_smooth_factor(int)));
    QObject::connect(main_window_ui.sharpen_button, SIGNAL(clicked()), this, SLOT(slot_sharpen_edge()));
    QObject::connect(main_window_ui.smooth_button, SIGNAL(clicked()), this, SLOT(slot_smooth_edge()));
    QObject::connect(main_window_ui.factor_slider_nmap, SIGNAL(valueChanged(int)), this, SLOT(slot_change_nmap_factor(int)));
    QObject::connect(main_window_ui.attenuation_slider_nmap, SIGNAL(valueChanged(int)), this, SLOT(slot_change_nmap_attenuation(int)));

    QObject::connect(main_window_ui.compute_dudv, SIGNAL(pressed()), this, SLOT(slot_compute_dudv()));
    QObject::connect(main_window_ui.factor_slider_dudv, SIGNAL(valueChanged(int)), this, SLOT(slot_change_dudv_nmap(int)));
    QObject::connect(main_window_ui.use_gpu, SIGNAL(clicked(bool)), this, SLOT(slot_use_gpgpu(bool)));

    QObject::connect(main_window_ui.bake_texture, SIGNAL(clicked()), this, SLOT(slot_cubemap_baking()));
    QObject::connect(main_window_ui.actionImport_3D_model, SIGNAL(triggered()), this, SLOT(slot_import_3DOBJ()));

    QObject::connect(main_window_ui.next_mesh_button, SIGNAL(clicked()), this, SLOT(slot_next_mesh()));
    QObject::connect(main_window_ui.previous_mesh_button, SIGNAL(clicked()), this, SLOT(slot_previous_mesh()));

    /*Renderer tab -> Post processing -> Camera*/
    QObject::connect(main_window_ui.gamma_slider, SIGNAL(valueChanged(int)), this, SLOT(slot_set_renderer_gamma_value(int)));
    QObject::connect(main_window_ui.exposure_slider, SIGNAL(valueChanged(int)), this, SLOT(slot_set_renderer_exposure_value(int)));
    QObject::connect(main_window_ui.reset_camera_button, SIGNAL(pressed()), this, SLOT(slot_reset_renderer_camera()));
    QObject::connect(main_window_ui.set_standard_post_p, SIGNAL(clicked()), this, SLOT(slot_set_renderer_no_post_process()));
    QObject::connect(main_window_ui.set_edge_post_p, SIGNAL(pressed()), this, SLOT(slot_set_renderer_edge_post_process()));
    QObject::connect(main_window_ui.set_sharpen_post_p, SIGNAL(pressed()), this, SLOT(slot_set_renderer_sharpen_post_process()));
    QObject::connect(main_window_ui.set_blurr_post_p, SIGNAL(pressed()), this, SLOT(slot_set_renderer_blurr_post_process()));

    /*Renderer tab -> Rasterization -> Polygon display*/
    QObject::connect(main_window_ui.rasterize_fill_button, SIGNAL(pressed()), this, SLOT(slot_set_rasterizer_fill()));
    QObject::connect(main_window_ui.rasterize_point_button, SIGNAL(pressed()), this, SLOT(slot_set_rasterizer_point()));
    QObject::connect(main_window_ui.rasterize_wireframe_button, SIGNAL(pressed()), this, SLOT(slot_set_rasterizer_wireframe()));
    QObject::connect(main_window_ui.rasterize_display_bbox_checkbox, SIGNAL(toggled(bool)), this, SLOT(slot_set_display_boundingbox(bool)));

    /*Renderer tab -> Lighting -> Point lights*/
    light_controller->connectAllSlots();

    /*Renderer tab -> UV editor -> mesh list*/
    QObject::connect(uv_editor_mesh_list, SIGNAL(itemSelectionChanged()), this, SLOT(slot_select_uv_editor_item()));

    /*Renderer tab -> Nova baking*/
    QObject::connect(main_window_ui.nova_bake_button, SIGNAL(pressed()), this, SLOT(slot_nova_start_bake()));
  }
}  // namespace controller