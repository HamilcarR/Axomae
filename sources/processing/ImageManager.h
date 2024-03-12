#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include "GenericException.h"
#include "Object3D.h"
#include "Rgb.h"

// TODO : Old code , Refactor

class SDL_Surface;

namespace axomae {

  struct max_colors;
  /***
   * @class ImageManager
   * @brief provides algorithms for image processing , like edge detection , greyscale conversion etc.
   *
   */
  class ImageManager {
   public:
    enum FILTER { FILTER_NULL = 0, GAUSSIAN_SMOOTH_3_3 = 0x01, GAUSSIAN_SMOOTH_5_5 = 0x02, BOX_BLUR = 0x03, SHARPEN = 0x04, UNSHARP_MASKING = 0x05 };
    static max_colors *get_colors_max_variation(SDL_Surface *image);
    static void compute_edge(SDL_Surface *greyscale_surface, uint8_t flag, uint8_t border_behaviour);
    static image::Rgb get_pixel_color(SDL_Surface *surface, int x, int y);
    static void print_pixel(uint32_t color);
    static void display_info_surface(SDL_Surface *image);
    static void set_pixel_color(SDL_Surface *image, int x, int y, uint32_t color);
    static void set_pixel_color(SDL_Surface *image, image::Rgb **pixel_array, int w, int h);
    static void set_greyscale_average(SDL_Surface *image, uint8_t factor);
    static void set_greyscale_luminance(SDL_Surface *image);
    static void set_contrast(SDL_Surface *image, int level);
    static void set_contrast(SDL_Surface *image);
    static void set_contrast_sigmoid(SDL_Surface *image, int treshold);
    static void compute_normal_map(SDL_Surface *surface, double strength, float attenuation);
    static void compute_dudv(SDL_Surface *surface, double factor);
    static void USE_GPU_COMPUTING() { gpu = true; }
    static void USE_CPU_COMPUTING() { gpu = false; }
    static bool USING_GPU() { return gpu; }
    static std::vector<uint8_t> project_uv_normals(const Object3D &object, int width, int height, bool tangent_space);
    static void smooth_image(SDL_Surface *surf, FILTER filter, const unsigned int smooth_iterations);
    static void sharpen_image(SDL_Surface *surf, FILTER filter, const unsigned int sharpen_iterations);

   private:
    ImageManager();
    ~ImageManager();

   private:
    static bool gpu;
  };

}  // namespace axomae

#endif
