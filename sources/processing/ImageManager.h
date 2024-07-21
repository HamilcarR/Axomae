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
   * @brief provides algorithms for image processing ,  edge detection , greyscale conversion etc.
   */
  class ImageManager {
   private:
    static bool gpu;

   private:
    ImageManager();
    ~ImageManager();

   public:
    enum FILTER { FILTER_NULL = 0, GAUSSIAN_SMOOTH_3_3 = 0x01, GAUSSIAN_SMOOTH_5_5 = 0x02, BOX_BLUR = 0x03, SHARPEN = 0x04, UNSHARP_MASKING = 0x05 };
    static max_colors *getColorsMaxVariations(SDL_Surface *image);
    static void computeEdge(SDL_Surface *greyscale_surface, uint8_t flag, uint8_t border_behaviour);
    static image::Rgb getPixelColor(SDL_Surface *surface, int x, int y);
    static void printPixel(uint32_t color);
    static void displayInfoSurface(SDL_Surface *image);
    static void setPixelColor(SDL_Surface *image, int x, int y, uint32_t color);
    static void setPixelColor(SDL_Surface *image, image::Rgb **pixel_array, int w, int h);
    static void setGrayscaleAverage(SDL_Surface *image, uint8_t factor);
    static void setGrayscaleLuminance(SDL_Surface *image);
    static void setContrast(SDL_Surface *image, int level);
    static void setContrast(SDL_Surface *image);
    static void setContrastSigmoid(SDL_Surface *image, int treshold);
    static void computeNormalMap(SDL_Surface *surface, double strength, float attenuation);
    static void computeDUDV(SDL_Surface *surface, double factor);
    static void useGPU() { gpu = true; }
    static void useCPU() { gpu = false; }
    static bool isUsingGPU() { return gpu; }
    static std::vector<uint8_t> projectUVNormals(const Object3D &object, int width, int height, bool tangent_space);
    static void smoothImage(SDL_Surface *surf, FILTER filter, unsigned int smooth_iterations);
    static void sharpenImage(SDL_Surface *surf, FILTER filter, unsigned int sharpen_iterations);
  };

}  // namespace axomae

#endif
