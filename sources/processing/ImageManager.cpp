#include "ImageManager.h"
#include "Kernel.cuh"
#include "Rgb.h"
#include "constants.h"
#include "project_macros.h"
#include <SDL2/SDL.h>
#include <assert.h>

/*Very Old code , to be refactored */

namespace axomae {

  using namespace image;

  bool ImageManager::gpu = false;
  static bool CHECK_IF_CUDA_AVAILABLE() {
    if (!ImageManager::isUsingGPU())
      return false;
    else
      return true;
  }

  template<typename T>
  static void replace_image(SDL_Surface *surface, T *image, unsigned int size, int bpp);
  ImageManager::ImageManager() {}
  ImageManager::~ImageManager() {}

  /**************************************************************************************************************/
  template<typename T>
  uint8_t truncate(T n) {
    if (static_cast<float>(n) <= 0.)
      return 0;
    else if (static_cast<float>(n) >= 255)
      return 255;
    else
      return static_cast<float>(n);
  }

  /**************************************************************************************************************/
  Rgb ImageManager::getPixelColor(SDL_Surface *surface, int x, int y) {
    int bpp = surface->format->BytesPerPixel;
    uint8_t *color = (uint8_t *)(surface->pixels) + x * bpp + y * surface->pitch;
    float red = 0, blue = 0, green = 0, alpha = 0;
    switch (bpp) {
      case 1: {
        red = *color >> 5 & 0x7;
        green = *color >> 2 & 0x7;
        blue = *color & 0x3;
      } break;
      case 2: {
        uint16_t colo16bits = *(uint16_t *)color;
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
          red = colo16bits >> 12 & 0xF;
          green = colo16bits >> 8 & 0XF;
          blue = colo16bits >> 4 & 0XF;
          alpha = colo16bits & 0XF;
        } else {
          alpha = colo16bits >> 12 & 0xF;
          blue = colo16bits >> 8 & 0XF;
          green = colo16bits >> 4 & 0XF;
          red = colo16bits & 0XF;
        }
      } break;
      case 3: {
        uint32_t colo24bits = *(uint32_t *)color;
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
          red = colo24bits >> 16 & 0XFF;
          green = colo24bits >> 8 & 0XFF;
          blue = colo24bits & 0XFF;
        } else {
          blue = colo24bits >> 16 & 0XFF;
          green = colo24bits >> 8 & 0XFF;
          red = colo24bits & 0XFF;
        }
      } break;
      case 4: {
        uint32_t colo32bits = *(uint32_t *)color;
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
          red = colo32bits >> 24 & 0XFF;
          green = colo32bits >> 16 & 0XFF;
          blue = colo32bits >> 8 & 0XFF;
          alpha = colo32bits & 0XFF;
        } else {
          alpha = colo32bits >> 24 & 0XFF;
          blue = colo32bits >> 16 & 0XFF;
          green = colo32bits >> 8 & 0XFF;
          red = colo32bits & 0XFF;
        }
      } break;
    }
    Rgb rgb = Rgb(static_cast<float>(red), static_cast<float>(green), static_cast<float>(blue), static_cast<float>(alpha));
    return rgb;
  }

  /**************************************************************************************************************/
  void ImageManager::setPixelColor(SDL_Surface *surface, int x, int y, uint32_t color) {
    // TODO : Add Tiling management when uv coordinates greater than limit
    int bpp = surface->format->BytesPerPixel;
    SDL_LockSurface(surface);
    Uint8 *pix = &((Uint8 *)surface->pixels)[x * bpp + y * surface->pitch];
    if (bpp == 1) {
      Uint8 *pixel = (Uint8 *)pix;
      *pixel = color;
    } else if (bpp == 2)
      *((Uint16 *)pix) = color;
    else if (bpp == 3) {
      if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
        ((Uint8 *)pix)[0] = color >> 16 & 0XFF;
        ((Uint8 *)pix)[1] = color >> 8 & 0XFF;
        ((Uint8 *)pix)[2] = color & 0XFF;
      } else {
        ((Uint8 *)pix)[0] = color & 0XFF;
        ((Uint8 *)pix)[1] = color >> 8 & 0XFF;
        ((Uint8 *)pix)[2] = color >> 16 & 0XFF;
      }
    } else
      *((Uint32 *)pix) = color;
    SDL_UnlockSurface(surface);
  }

  /**************************************************************************************************************/

  void ImageManager::printPixel(uint32_t color) {
    uint8_t red = color >> 24 & 0XFF;
    uint8_t green = color >> 16 & 0XFF;
    uint8_t blue = color >> 8 & 0XFF;
    uint8_t alpha = color & 0XFF;
    std::cout << "red : " << std::to_string(red) << "\n"
              << "green : " << std::to_string(green) << "\n"
              << "blue : \n"
              << std::to_string(blue) << "\n"
              << "alpha : " << std::to_string(alpha) << "\n";
  }

  /**************************************************************************************************************/

  void ImageManager::displayInfoSurface(SDL_Surface *image) {
    std::cout << "Bytes per pixel : " << std::to_string(image->format->BytesPerPixel) << "\n";
    std::cout << "Padding on X : " << std::to_string(image->format->padding[0]) << "\n";
    std::cout << "Padding on Y : " << std::to_string(image->format->padding[1]) << "\n";
  }

  /**************************************************************************************************************/

  void ImageManager::setPixelColor(SDL_Surface *surface, Rgb **arrayc, int w, int h) {
    for (int i = 0; i < w; i++)
      for (int j = 0; j < h; j++)
        setPixelColor(surface, i, j, arrayc[i][j].rgb_to_int());
  }

  /**************************************************************************************************************/
  void ImageManager::setGrayscaleAverage(SDL_Surface *image, uint8_t factor) {
    assert(factor > 0);
    assert(image != nullptr);
    if (CHECK_IF_CUDA_AVAILABLE())
      GPU_compute_greyscale(image, false);
    else
      for (int i = 0; i < image->w; i++)
        for (int j = 0; j < image->h; j++) {
          Rgb rgb = getPixelColor(image, i, j);
          rgb.red = (rgb.red + rgb.blue + rgb.green) / factor;
          rgb.green = rgb.red;
          rgb.blue = rgb.red;
          uint32_t gray = rgb.rgb_to_int();
          setPixelColor(image, i, j, gray);
        }
  }

  /**************************************************************************************************************/
  static void replace_image(SDL_Surface *surface, uint8_t *image) {
    int bpp = surface->format->BytesPerPixel;
    SDL_LockSurface(surface);
    if (bpp == 1) {
      for (int i = 0; i < surface->w; i++)
        for (int j = 0; j < surface->h; j++)
          ((Uint8 *)surface->pixels)[i * bpp + j * surface->pitch] = image[i * bpp + j * surface->pitch];
      delete[] static_cast<uint8_t *>(image);
    } else
      std::cerr << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
    SDL_UnlockSurface(surface);
  }

  /**************************************************************************************************************/
  static void replace_image(SDL_Surface *surface, uint16_t *image) {
    int bpp = surface->format->BytesPerPixel;
    SDL_LockSurface(surface);
    if (bpp == 2) {
      for (int i = 0; i < surface->w; i++)
        for (int j = 0; j < surface->h; j++)
          ((Uint16 *)surface->pixels)[i * bpp + j * surface->pitch] = image[i * bpp + j * surface->pitch];
      delete[] static_cast<uint16_t *>(image);
    } else
      std::cerr << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
    SDL_UnlockSurface(surface);
  }

  /**************************************************************************************************************/

  static void replace_image(SDL_Surface *surface, uint32_t *image) {
    int bpp = surface->format->BytesPerPixel;
    int pitch = surface->pitch;
    SDL_LockSurface(surface);
    if (bpp == 4) {
      for (int i = 0; i < surface->w; i++)
        for (int j = 0; j < surface->h; j++)
          ((Uint32 *)surface->pixels)[i * bpp + j * surface->pitch] = image[i * bpp + j * surface->pitch];
      delete[] static_cast<uint32_t *>(image);

    } else if (bpp == 3)
      for (int i = 0; i < surface->w; i++)
        for (int j = 0; j < surface->h; j++)
          ((Uint32 *)surface->pixels)[i * bpp + j * surface->pitch] = image[i * bpp + j * surface->pitch];
    else
      std::cerr << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
    SDL_UnlockSurface(surface);
  }

  /**************************************************************************************************************/
  void ImageManager::setGrayscaleLuminance(SDL_Surface *image) {
    bool cuda = CHECK_IF_CUDA_AVAILABLE();
    std::clock_t clock;
    if (cuda) {
      GPU_compute_greyscale(image, true);
    } else {
      assert(image != nullptr);
      for (int i = 0; i < image->w; i++)
        for (int j = 0; j < image->h; j++) {
          Rgb rgb = getPixelColor(image, i, j);
          rgb.red = floor(rgb.red * 0.3 + rgb.blue * 0.11 + rgb.green * 0.59);
          rgb.green = rgb.red;
          rgb.blue = rgb.red;
          uint32_t gray = rgb.rgb_to_int();
          setPixelColor(image, i, j, gray);
        }
    }
  }

  /**************************************************************************************************************/
  template<typename T>
  static T compute_kernel_pixel(Rgb **data, const T kernel[3][3], int i, int j, uint8_t flag) {
    if (flag == AXOMAE_RED) {
      return data[i - 1][j - 1].red * kernel[0][0] + data[i][j - 1].red * kernel[0][1] + data[i + 1][j - 1].red * kernel[0][2] +
             data[i - 1][j].red * kernel[1][0] + data[i][j].red * kernel[1][1] + data[i + 1][j].red * kernel[1][2] +
             data[i - 1][j + 1].red * kernel[2][0] + data[i][j + 1].red * kernel[2][1] + data[i + 1][j + 1].red * kernel[2][2];
    } else if (flag == AXOMAE_BLUE) {
      return data[i - 1][j - 1].blue * kernel[0][0] + data[i][j - 1].blue * kernel[0][1] + data[i + 1][j - 1].blue * kernel[0][2] +
             data[i - 1][j].blue * kernel[1][0] + data[i][j].blue * kernel[1][1] + data[i + 1][j].blue * kernel[1][2] +
             data[i - 1][j + 1].blue * kernel[2][0] + data[i][j + 1].blue * kernel[2][1] + data[i + 1][j + 1].blue * kernel[2][2];
    } else {
      return data[i - 1][j - 1].green * kernel[0][0] + data[i][j - 1].green * kernel[0][1] + data[i + 1][j - 1].green * kernel[0][2] +
             data[i - 1][j].green * kernel[1][0] + data[i][j].green * kernel[1][1] + data[i + 1][j].green * kernel[1][2] +
             data[i - 1][j + 1].green * kernel[2][0] + data[i][j + 1].green * kernel[2][1] + data[i + 1][j + 1].green * kernel[2][2];
    }
  }
  template<typename T>
  static Rgb compute_generic_kernel_pixel(const Rgb **data, const unsigned int size, const T **kernel, const unsigned int x, const unsigned int y) {
    float r = 0, g = 0, b = 0;
    uint8_t middle = std::floor(size / 2);
    for (unsigned int i = 0; i < size; i++)
      for (unsigned int j = 0; j < size; j++) {
        float kernel_val = kernel[i][j];
        int new_i = i - middle, new_j = j - middle;
        r += data[x + new_i][y + new_j].red * kernel_val;
        g += data[x + new_i][y + new_j].green * kernel_val;
        b += data[x + new_i][y + new_j].blue * kernel_val;
      }
    return Rgb(r, g, b);
  }

  /**************************************************************************************************************/
  void ImageManager::computeEdge(SDL_Surface *surface, uint8_t flag, uint8_t border) {
    bool cuda = CHECK_IF_CUDA_AVAILABLE();
    if (cuda)
      GPU_compute_height(surface, flag, border);
    else {
      // TODO : use multi threading for initialization and greyscale computing
      /*to avoid concurrent access on image*/

      std::cout << "CUDA : " << cuda << "\n";

      int w = surface->w;
      int h = surface->h;
      // thread this :
      Rgb **data = new Rgb *[w];
      for (int i = 0; i < w; i++)
        data[i] = new Rgb[h];
      float max_red = 0, max_blue = 0, max_green = 0;
      float min_red = 0, min_blue = 0, min_green = 0;
      for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
          Rgb rgb = getPixelColor(surface, i, j);
          max_red = (rgb.red >= max_red) ? rgb.red : max_red;
          max_green = (rgb.green >= max_green) ? rgb.green : max_green;
          max_blue = (rgb.blue >= max_blue) ? rgb.blue : max_blue;
          min_red = (rgb.red < min_red) ? rgb.red : min_red;
          min_green = (rgb.green < min_green) ? rgb.green : min_green;
          min_blue = (rgb.blue < min_blue) ? rgb.blue : min_blue;
          data[i][j].red = rgb.red;
          data[i][j].blue = rgb.blue;
          data[i][j].green = rgb.green;
        }
      }
      int arr_h[3][3];
      int arr_v[3][3];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if (flag == AXOMAE_USE_SOBEL) {
            arr_v[i][j] = sobel_mask_vertical[i][j];
            arr_h[i][j] = sobel_mask_horizontal[i][j];
          } else if (flag == AXOMAE_USE_PREWITT) {
            arr_h[i][j] = prewitt_mask_horizontal[i][j];
            arr_v[i][j] = prewitt_mask_vertical[i][j];
          } else if (flag == AXOMAE_USE_SCHARR) {
            arr_h[i][j] = scharr_mask_horizontal[i][j];
            arr_v[i][j] = scharr_mask_vertical[i][j];
          }
        }
      }
      for (int i = 1; i < w - 1; i++) {
        for (int j = 1; j < h - 1; j++) {
          if (border == AXOMAE_REPEAT) {
            float setpix_h_red = 0, setpix_v_red = 0, setpix_h_green = 0, setpix_v_green = 0, setpix_v_blue = 0, setpix_h_blue = 0;
            setpix_h_red = compute_kernel_pixel(data, arr_h, i, j, AXOMAE_RED);
            setpix_v_red = compute_kernel_pixel(data, arr_v, i, j, AXOMAE_RED);
            setpix_h_green = compute_kernel_pixel(data, arr_h, i, j, AXOMAE_GREEN);
            setpix_v_green = compute_kernel_pixel(data, arr_v, i, j, AXOMAE_GREEN);
            setpix_h_blue = compute_kernel_pixel(data, arr_h, i, j, AXOMAE_BLUE);
            setpix_v_blue = compute_kernel_pixel(data, arr_v, i, j, AXOMAE_BLUE);
            setpix_v_red = normalize(max_red, min_red, setpix_v_red);
            setpix_h_red = normalize(max_red, min_red, setpix_h_red);
            setpix_v_green = normalize(max_green, min_green, setpix_v_green);
            setpix_h_green = normalize(max_green, min_green, setpix_h_green);
            setpix_v_blue = normalize(max_blue, min_blue, setpix_v_blue);
            setpix_h_blue = normalize(max_blue, min_blue, setpix_h_blue);
            float r = magnitude(setpix_v_red, setpix_h_red);
            float g = magnitude(setpix_v_green, setpix_h_green);
            float b = magnitude(setpix_v_blue, setpix_h_blue);
            Rgb rgb = Rgb(r, g, b, 0);
            setPixelColor(surface, i, j, rgb.rgb_to_int());
          }
        }
      }
      auto del = std::async(std::launch::async, [data, w]() {
        for (int i = 0; i < w; i++)
          delete[] data[i];
        delete[] data;
      });
    }
  }

  /**************************************************************************************************************/
  max_colors *ImageManager::getColorsMaxVariations(SDL_Surface *image) {
    max_colors *max_min = new max_colors;
    const int INT_MAXX = 0;
    int max_red = 0, max_green = 0, max_blue = 0, min_red = INT_MAX, min_blue = INT_MAX, min_green = INT_MAXX;
    for (int i = 0; i < image->w; i++) {
      for (int j = 0; j < image->h; j++) {
        Rgb rgb = getPixelColor(image, i, j);
        max_red = (rgb.red >= max_red) ? rgb.red : max_red;
        max_green = (rgb.green >= max_green) ? rgb.green : max_green;
        max_blue = (rgb.blue >= max_blue) ? rgb.blue : max_blue;
        min_red = (rgb.red < min_red) ? rgb.red : min_red;
        min_green = (rgb.green < min_green) ? rgb.green : min_green;
        min_blue = (rgb.blue < min_blue) ? rgb.blue : min_blue;
      }
    }
    max_min->max_rgb[0] = max_red;
    max_min->max_rgb[1] = max_green;
    max_min->max_rgb[2] = max_blue;
    max_min->min_rgb[0] = min_red;
    max_min->min_rgb[1] = min_green;
    max_min->min_rgb[2] = min_blue;
    return max_min;
  }

  /**************************************************************************************************************/
  void ImageManager::setContrast(SDL_Surface *image, int level) {
    double correction_factor = (259 * (level + 255)) / (255 * (259 - level));
    max_colors *maxmin = getColorsMaxVariations(image);
    for (int i = 0; i < image->w; i++) {
      for (int j = 0; j < image->h; j++) {
        Rgb col = getPixelColor(image, i, j);
        col.red = floor(truncate(correction_factor * (col.red - 128) + 128));
        col.green = floor(truncate(correction_factor * (col.green - 128) + 128));
        col.blue = floor(truncate(correction_factor * (col.blue - 128) + 128));
        col.alpha = 0;
        setPixelColor(image, i, j, col.rgb_to_int());
      }
    }
    delete maxmin;
  }

  /**************************************************************************************************************/
  void ImageManager::setContrast(SDL_Surface *image) {
    const int val = 200;
    for (int i = 0; i < image->w; i++) {
      for (int j = 0; j < image->h; j++) {
        Rgb col = getPixelColor(image, i, j);
        col.red = col.red <= val ? 0 : 255;
        col.blue = col.blue <= val ? 0 : 255;
        col.green = col.green <= val ? 0 : 255;
        setPixelColor(image, i, j, col.rgb_to_int());
      }
    }
  }

  /***************************************************************************************************************/
  /*TODO : Contrast image enhancement */
  void ImageManager::setContrastSigmoid(SDL_Surface *image, int threshold) {
    for (int i = 0; i < image->w; i++) {
      for (int j = 0; j < image->h; j++) {
        Rgb color = getPixelColor(image, i, j);
        // RGB normalized = normalize_0_1(color);
      }
    }
  }
  /***************************************************************************************************************/
  constexpr double radiant_to_degree(double rad) { return rad * 180.f / M_PI; }

  /**************************************************************************************************************/
  constexpr double get_pixel_height(double color_component) { return (255 - color_component); }

  /**************************************************************************************************************/
  void ImageManager::computeNormalMap(SDL_Surface *surface, double fact, float attenuation) {
    bool cuda = CHECK_IF_CUDA_AVAILABLE();
    if (cuda)
      GPU_compute_normal(surface, fact, AXOMAE_REPEAT);
    else {
      int height = surface->h;
      int width = surface->w;
      Rgb **data = new Rgb *[width];
      for (int i = 0; i < width; i++)
        data[i] = new Rgb[height];
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++)
          data[i][j] = getPixelColor(surface, i, j);
      }
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
          int x, y, a, b;
          x = (i == 0) ? i + 1 : i - 1;
          y = (j == 0) ? j + 1 : j - 1;
          a = (i == width - 1) ? width - 2 : i + 1;
          b = (j == height - 1) ? height - 2 : j + 1;
          double col_left = data[i][y].green;
          double col_right = data[i][b].green;
          double col_up = data[x][j].green;
          double col_down = data[a][j].green;
          double col_up_right = data[x][b].green;
          double col_up_left = data[x][y].green;
          double col_down_left = data[a][y].green;
          double col_down_right = data[a][b].green;
          float atten = attenuation;
          double dx = atten * (fact * (col_right - col_left) / 255);
          double dy = atten * (fact * (col_up - col_down) / 255);
          double ddx = atten * (fact * (col_up_right - col_down_left) / 255);
          double ddy = atten * (fact * (col_up_left - col_down_right) / 255);
          auto Nx = normalize(-1, 1, lerp(dy, ddy, 0.5));
          auto Ny = normalize(-1, 1, lerp(dx, ddx, 0.5));
          auto Nz = 255.0;  // the normal vector
          Rgb col = Rgb(floor(truncate(Nx)), floor(truncate(Ny)), Nz);
          setPixelColor(surface, i, j, col.rgb_to_int());
        }
      }
    }
  }

  /***************************************************************************************************************/
  void ImageManager::computeDUDV(SDL_Surface *surface, double factor) {
    int height = surface->h;
    int width = surface->w;
    Rgb **data = new Rgb *[width];
    for (int i = 0; i < width; i++)
      data[i] = new Rgb[height];
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++)
        data[i][j] = getPixelColor(surface, i, j);
    }
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        int x, y, a, b;
        x = (i == 0) ? i + 1 : i - 1;
        y = (j == 0) ? j + 1 : j - 1;
        a = (i == width - 1) ? width - 2 : i + 1;
        b = (j == height - 1) ? height - 2 : j + 1;
        Rgb col_left = data[i][y];
        Rgb col_right = data[i][b];
        Rgb col_up = data[x][j];
        Rgb col_down = data[a][j];
        Rgb col_up_right = data[x][b];
        Rgb col_up_left = data[x][y];
        Rgb col_down_left = data[a][y];
        Rgb col_down_right = data[a][b];
        double atten = 0.8;
        double dx_red = atten * (factor * (col_left.red - col_right.red) / 255);
        double dx_green = atten * (factor * (col_left.green - col_right.green) / 255);
        double dy_red = atten * (factor * (col_up.red - col_down.red) / 255);
        double dy_green = atten * (factor * (col_up.green - col_down.green) / 255);
        double ddx_green = atten * (factor * (col_up_right.green - col_down_left.green) / 255);
        double ddy_green = atten * (factor * (col_up_left.green - col_down_right.green) / 255);
        double ddx_red = atten * (factor * (col_up_right.red - col_down_left.red) / 255);
        double ddy_red = atten * (factor * (col_up_left.red - col_down_right.red) / 255);
        auto red_var = normalize(-1, 1, lerp(dx_red + dy_red, ddx_red + ddy_red, 0.5));
        auto green_var = normalize(-1, 1, lerp(dx_green + dy_green, ddx_green + ddy_green, 0.5));
        Rgb col = Rgb(truncate(red_var), truncate(green_var), 0.0);
        setPixelColor(surface, i, j, col.rgb_to_int());
      }
    }
  }

  /**************************************************************************************************************/
  void ImageManager::smoothImage(SDL_Surface *surface, FILTER filter, const unsigned int factor) {
    if (surface != nullptr) {
      int height = surface->h;
      int width = surface->w;
      Rgb **data = new Rgb *[width];
      for (int i = 0; i < width; i++)
        data[i] = new Rgb[height];
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++)
          data[i][j] = getPixelColor(surface, i, j);
      }
      int n = 0;
      void *convolution_kernel = nullptr;
      switch (filter) {
        case GAUSSIAN_SMOOTH_3_3:
          convolution_kernel = (void *)gaussian_blur_3_3;
          n = 3;
          break;
        case GAUSSIAN_SMOOTH_5_5:
          convolution_kernel = (void *)gaussian_blur_5_5;
          n = 5;
          break;
        case BOX_BLUR:
          convolution_kernel = (void *)box_blur;
          n = 3;
          break;
        default:
          convolution_kernel = nullptr;
          n = 0;
          break;
      }
      float **blur = new float *[n];
      static unsigned int kernel_index = 0;
      for (int i = 0; i < n; i++)
        blur[i] = new float[n];
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++, kernel_index++)
          blur[i][j] = static_cast<float *>(convolution_kernel)[kernel_index];
      kernel_index = 0;
      unsigned int middle = std::floor(n / 2);
      for (unsigned int i = middle; i < width - middle; i++) {
        for (unsigned int j = middle; j < height - middle; j++) {
          Rgb col;
          col = compute_generic_kernel_pixel(const_cast<const Rgb **>(data), n, const_cast<const float **>(blur), i, j);
          setPixelColor(surface, i, j, col.rgb_to_int());
        }
      }
      for (int i = 0; i < n; i++)
        delete[] blur[i];
      delete[] blur;

      if (factor != 0)
        smoothImage(surface, filter, factor - 1);
    }
  }
  /**************************************************************************************************************/
  void ImageManager::sharpenImage(SDL_Surface *surface, FILTER filter, const unsigned int factor) {
    if (surface != nullptr) {
      int height = surface->h;
      int width = surface->w;
      Rgb **data = new Rgb *[width];
      for (int i = 0; i < width; i++)
        data[i] = new Rgb[height];
      for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
          data[i][j] = getPixelColor(surface, i, j);

      int n = 0;
      void *convolution_kernel = nullptr;
      switch (filter) {
        case SHARPEN:
          convolution_kernel = (void *)sharpen_kernel;
          n = 3;
          break;
        case UNSHARP_MASKING:
          convolution_kernel = (void *)unsharp_masking;
          n = 5;
          break;
        default:
          convolution_kernel = nullptr;
          n = 0;
          break;
      }
      float **sharp = new float *[n];
      static unsigned int kernel_index = 0;
      for (int i = 0; i < n; i++)
        sharp[i] = new float[n];
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++, kernel_index++)
          sharp[i][j] = static_cast<float *>(convolution_kernel)[kernel_index];
      kernel_index = 0;
      unsigned int middle = std::floor(n / 2);
      for (unsigned int i = middle; i < width - middle; i++) {
        for (unsigned int j = middle; j < height - middle; j++) {
          Rgb col;
          col = compute_generic_kernel_pixel(const_cast<const Rgb **>(data), n, const_cast<const float **>(sharp), i, j);
          col.clamp();
          setPixelColor(surface, i, j, col.rgb_to_int());
        }
      }
      for (int i = 0; i < n; i++)
        delete[] sharp[i];
      delete[] sharp;
    }
  }

  // end namespace
}  // namespace axomae
