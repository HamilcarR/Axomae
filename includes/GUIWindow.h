#ifndef GUIWINDOW_H
#define GUIWINDOW_H

#include <QtWidgets/qapplication.h>
#include <QtWidgets/qmainwindow.h>
#include <QtWidgets/qpushbutton.h>

#include "../Form Files/ui_test.h"
#include "Config.h"
#include "LightControllerUI.h"
#include "Renderer.h"
#include "SceneListView.h"
#include "SceneSelector.h"
#include "Window.h"
#include "constants.h"
#include "utils_3D.h"

/**
 * @file GUIWindow.h
 * UI layout
 *
 */

namespace gui {
  enum IMAGETYPE : unsigned { GREYSCALE_LUMI = 1, HEIGHT = 2, NMAP = 3, DUDV = 4, ALBEDO = 5, GREYSCALE_AVG = 6, PROJECTED_NMAP = 7, INVALID = 8 };
}
namespace axomae {
  class HeapManagement;
  class ImageImporter;
  template<typename T>
  struct image_type;
  class Controller : public QMainWindow {
    Q_OBJECT
   public:
    Controller(QWidget *parent = nullptr);
    ~Controller();
    void setApplicationConfig(const std::string &config_string);
    Ui::MainWindow &getUi() {
      return _UI;
    }
    static HeapManagement *_MemManagement;
    static SDL_Surface *copy_surface(SDL_Surface *surface);

    /* SLOTS */
   public slots:

    /**
     * @brief Loads an image from disk
     *
     * @return true if operation successful
     * @return false if there was a problem
     */
    bool import_image();

    /**
     * @brief Loads a 3D GLB object
     *
     * @return true if operation successful
     * @return false if there was a problem
     */
    bool import_3DOBJ();

    /**
     * @brief Opens an Axomae project
     *
     * @return true
     * @return false
     */
    bool open_project();

    /**
     * @brief Saves an Axomae project
     *
     * @return true
     * @return false
     */
    bool save_project();

    /**
     * @brief Saves the baked textures on the disk
     *
     * @return true
     * @return false
     */
    bool save_image();

    /**
     * @brief Computes the greyscale average of the previous texture in the stack
     *
     * @return true
     * @return false
     */
    bool greyscale_average();

    /**
     * @brief Computes the greyscale luminance of the previous texture in the stack
     *
     * @return true
     * @return false
     */
    bool greyscale_luminance();

    /**
     * @brief Computes the Scharr edge detection algorithm of the previous texture in the stack
     *
     */
    void use_scharr();

    /**
     * @brief Computes the Prewitt version of the edge detection
     *
     */
    void use_prewitt();

    /**
     * @brief Computes the Sobel edge detection algorithm
     *
     */
    void use_sobel();

    /**
     * @brief Enable Cuda computation inside of CPU side
     *
     * @param checked True if the option has been enabled
     */
    void use_gpgpu(bool checked);

    /**
     * @brief Computes normals in object space
     *
     */
    void use_object_space();

    /**
     * @brief Computes normals in tangent space by projecting them from a 3D object to a texture
     *
     */
    void use_tangent_space();

    /**
     * @brief Chage depth factor from the edge textures , allowing more emphasized normal vectors
     *
     * @param factor depth factor
     */
    void change_nmap_factor(int factor);

    /**
     * @brief Provides smoothing to the normal vectors on the texture
     *
     * @param atten Attenuation value
     */
    void change_nmap_attenuation(int atten);

    /**
     * @brief Computes the DUDV texture from the normal map
     *
     */
    void compute_dudv();

    /**
     * @brief Provides a clearer topography of the texture by changing the depth factor
     *
     * @param factor Depth factor
     */
    void change_dudv_nmap(int factor);

    /**
     * @brief Normal map traditionnal baking method
     *
     */
    void compute_projection();

    /**
     * @brief Skip to next mesh in the mesh selector
     *
     */
    void next_mesh();

    /**
     * @brief Skip to the previous mesh in the mesh selector
     *
     */
    void previous_mesh();

    /**
     * @brief Project the normals of the mesh on his UVs
     *
     */
    void project_uv_normals();

    /**
     * @brief Provides smoothing for the edge detection algorithm
     *
     */
    void smooth_edge();

    /**
     * @brief Sharpens the edges on the edge detection
     *
     */
    void sharpen_edge();

    /**
     * @brief Undo the current modification
     *
     */
    void undo();

    /**
     * @brief Redo the previous modification
     *
     */
    void redo();

    /**
     * @brief Set the gamma value on the renderer
     *
     * @param gamma
     */
    void set_renderer_gamma_value(int gamma);

    /**
     * @brief Set the exposure value on the renderer
     *
     * @param exposure
     */
    void set_renderer_exposure_value(int exposure);

    /**
     * @brief Resets the scene camera to default
     *
     *
     */
    void reset_renderer_camera();

    /**
     * @brief disable post processing
     *
     */
    void set_renderer_no_post_process();

    /**
     * @brief Post processing to edge contours
     *
     */
    void set_renderer_edge_post_process();

    /**
     * @brief Post processing to sharpen
     *
     */
    void set_renderer_sharpen_post_process();

    /**
     * @brief Post processing to blurr
     *
     */
    void set_renderer_blurr_post_process();

    /**
     * @brief Set the rasterizer point object
     *
     */
    void set_rasterizer_point();

    /**
     * @brief Set the rasterizer fill object
     *
     */
    void set_rasterizer_fill();

    /**
     * @brief Set the rasterizer wireframe object
     *
     */
    void set_rasterizer_wireframe();

    /**
     * @brief Set the display boundingbox object
     *
     * @param display
     */
    void set_display_boundingbox(bool display);

   protected slots:
    /**
     * @brief Updates UI slider according to the factor
     *
     * @param factor change factor
     */
    void update_smooth_factor(int factor);

   private:
    void connect_all_slots();
    QGraphicsView *get_corresponding_view(gui::IMAGETYPE image);
    SDL_Surface *get_corresponding_session_pointer(gui::IMAGETYPE image);
    bool set_corresponding_session_pointer(image_type<SDL_Surface> *image_type_pointer);
    void display_image(SDL_Surface *surf, gui::IMAGETYPE image, bool save_in_heap);

   private:
    Ui::MainWindow _UI;
    GLViewer *viewer_3d;
    Window *_window;
    ImageImporter *_importer;
    ApplicationConfig configuration;
    SceneListView *renderer_scene_list;
    std::unique_ptr<LightController> light_controller;
  };

}  // namespace axomae

#endif
