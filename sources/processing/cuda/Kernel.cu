#include "Kernel.cuh"
#include <cmath>
#include <sstream>

// Old code , tb refactored

namespace axomae {

  AX_DEVICE_ONLY const static bool isbigEndian = SDL_BIG_ENDIAN == SDL_BYTEORDER;
  AX_DEVICE_ONLY static uint32_t max_int_rgb = 0;
  AX_DEVICE_ONLY static uint32_t min_int_rgb = UINT32_MAX;

  struct Triplet {
    int x;
    int y;
    int z;
  };

  struct DEVICE_RGB {
   public:
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    AX_DEVICE_ONLY void operator=(DEVICE_RGB rgb) {
      this->r = rgb.r;
      this->g = rgb.g;
      this->b = rgb.b;
      this->a = rgb.a;
    }
    std::string string() { return std::to_string(r) + "   " + std::to_string(g) + "  " + std::to_string(b) + "\n"; }
    AX_DEVICE_ONLY DEVICE_RGB operator+(DEVICE_RGB rgb) { return {uint8_t(r + rgb.r), uint8_t(g + rgb.g), uint8_t(b + rgb.b), uint8_t(a + rgb.a)}; }
    template<typename T>
    AX_DEVICE_ONLY DEVICE_RGB operator+(T rgb) {
      return {r + rgb, g + rgb, b + rgb, a + rgb};
    }
    AX_DEVICE_ONLY DEVICE_RGB operator*(DEVICE_RGB rgb) { return {uint8_t(r * rgb.r), uint8_t(g * rgb.g), uint8_t(b * rgb.b), uint8_t(a * rgb.a)}; }
    template<typename T>
    AX_DEVICE_ONLY DEVICE_RGB operator*(T value) {
      return {uint8_t(r * value), uint8_t(g * value), uint8_t(b * value), uint8_t(a * value)};
    }
    AX_DEVICE_ONLY DEVICE_RGB normalize_rgb(DEVICE_RGB max, DEVICE_RGB min) {
      uint8_t n_red = normalize(max.r, min.r, r);
      uint8_t n_green = normalize(max.g, min.g, g);
      uint8_t n_blue = normalize(max.b, min.b, b);
      return {n_red, n_green, n_blue, 0};
    }
    /*compute the magnitude between to rgb values*/
    AX_DEVICE_ONLY DEVICE_RGB magnitude_rgb(DEVICE_RGB horizontal, DEVICE_RGB vertical) {
      DEVICE_RGB rgb;
      rgb.r = (uint8_t)magnitude(vertical.r, horizontal.r);
      rgb.g = (uint8_t)magnitude(vertical.g, horizontal.g);
      rgb.b = (uint8_t)magnitude(vertical.b, horizontal.b);
      rgb.a = (uint8_t)magnitude(vertical.a, horizontal.a);
      return rgb;
    }
    AX_DEVICE_ONLY void print() { printf("%i %i %i\n", r, g, b); }
  };

  class SDLSurfParam {
   public:
    unsigned int width;
    unsigned int height;
    int bpp;
    int pitch;
    void *data;
    SDLSurfParam(SDL_Surface *im) {
      width = im->w;
      height = im->h;
      bpp = im->format->BytesPerPixel;
      pitch = im->pitch;
      data = im->pixels;
    }
    SDLSurfParam() {}
    size_t getByteSize() { return height * pitch; }
  };
  template<typename T>
  struct custom_convolution_kernel {
    T *array;
    uint8_t size_w;
    uint8_t size_h;
  };

  /*device*/
  /*********************************************************************************************************************************************/
  AX_DEVICE_CALLABLE uint32_t rgb_to_int(DEVICE_RGB val) {
    uint32_t value = (isbigEndian) ? val.a | (val.b << 8) | (val.g << 16) | (val.r << 24) : val.r | (val.g << 8) | (val.b << 16) | (val.a << 24);
    return value;
  }
  AX_DEVICE_ONLY void initialize_2D_array(uint32_t *array, int size_w, int size_h) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    array[i * size_w + j] = 0;
  }
  AX_DEVICE_ONLY DEVICE_RGB compute_greyscale(DEVICE_RGB rgb, const bool luminance) {
    DEVICE_RGB ret;
    if (luminance) {
      ret.r = rgb.r * 0.3 + rgb.g * 0.59 + rgb.b * 0.11;
      ret.g = ret.r;
      ret.b = ret.r;
    } else {
      ret.r = (int)((rgb.r + rgb.b + rgb.g) / 3);
      ret.g = ret.r;
      ret.b = ret.r;
    }
    return ret;
  }
  AX_DEVICE_ONLY DEVICE_RGB int_to_rgb(uint8_t *pixel_value, const int bpp) {
    DEVICE_RGB rgb = {0, 0, 0, 0};
    if (bpp == 4) {
      if (isbigEndian) {
        rgb.r = *pixel_value >> 24 & 0xFF;
        rgb.g = *pixel_value >> 16 & 0xFF;
        rgb.b = *pixel_value >> 8 & 0xFF;
        rgb.a = *pixel_value & 0xFF;
      } else {
        rgb.a = *pixel_value >> 24 & 0xFF;
        rgb.b = *pixel_value >> 16 & 0xFF;
        rgb.g = *pixel_value >> 8 & 0xFF;
        rgb.r = *pixel_value & 0xFF;
      }

    } else if (bpp == 3) {
      if (isbigEndian) {
        rgb.r = pixel_value[0];
        rgb.g = pixel_value[1];
        rgb.b = pixel_value[2];
        rgb.a = 0;
      } else {
        rgb.b = pixel_value[0];
        rgb.g = pixel_value[1];
        rgb.r = pixel_value[2];
        rgb.a = 0;
      }

    } else if (bpp == 2) {
      if (isbigEndian) {
        rgb.r = *pixel_value >> 12 & 0xF;
        rgb.g = *pixel_value >> 8 & 0XF;
        rgb.b = *pixel_value >> 4 & 0XF;
        rgb.a = *pixel_value & 0XF;
      } else {
        rgb.a = *pixel_value >> 12 & 0xF;
        rgb.b = *pixel_value >> 8 & 0XF;
        rgb.g = *pixel_value >> 4 & 0XF;
        rgb.r = *pixel_value & 0XF;
      }
    } else if (bpp == 1) {
      if (isbigEndian) {
        rgb.r = *pixel_value >> 5 & 0X7;
        rgb.g = *pixel_value >> 2 & 0X7;
        rgb.b = *pixel_value & 0X3;
        rgb.a = 0;
      } else {
        rgb.b = *pixel_value >> 5 & 0X7;
        rgb.g = *pixel_value >> 2 & 0X7;
        rgb.r = *pixel_value & 0X3;
        rgb.a = 0;
      }
    }
    return rgb;
  }
  AX_DEVICE_ONLY void set_pixel_color(uint8_t *pixel_value, DEVICE_RGB rgb, const int bpp) {
    uint32_t toInt = rgb_to_int(rgb);
    if (bpp == 4)
      *(uint32_t *)(pixel_value) = toInt;
    else if (bpp == 3) {
      if (isbigEndian) {
        ((uint8_t *)pixel_value)[0] = toInt >> 16 & 0xFF;
        ((uint8_t *)pixel_value)[1] = toInt >> 8 & 0xFF;
        ((uint8_t *)pixel_value)[2] = toInt & 0xFF;
      } else {
        ((uint8_t *)pixel_value)[0] = toInt & 0xFF;
        ((uint8_t *)pixel_value)[1] = toInt >> 8 & 0xFF;
        ((uint8_t *)pixel_value)[2] = toInt >> 16 & 0xFF;
      }
    } else if (bpp == 2)
      *((uint16_t *)pixel_value) = toInt;
    else
      *pixel_value = toInt;
  }
  AX_DEVICE_ONLY DEVICE_RGB get_pixel_value_at(uint8_t *pixel, int i, int j, const int bpp, int pitch) {
    uint8_t *p = (uint8_t *)(pixel) + i * bpp + j * pitch;
    DEVICE_RGB A = int_to_rgb(p, bpp);
    return A;
  }
  struct convolution_directions {
    DEVICE_RGB vertical;
    DEVICE_RGB horizontal;
  };

  // TODO : case kernel < 0
  AX_DEVICE_ONLY convolution_directions compute_convolution(uint8_t *pixel,
                                                      const int bpp,
                                                      int pitch,
                                                      const int h_kernel[KERNEL_SIZE][KERNEL_SIZE],
                                                      const int v_kernel[KERNEL_SIZE][KERNEL_SIZE],
                                                      uint8_t border_flag) {
    DEVICE_RGB center = get_pixel_value_at(pixel, 0, 0, bpp, pitch);
    DEVICE_RGB west = get_pixel_value_at(pixel, 0, -1, bpp, pitch);  // here : if threadIdx.y = 0 bug
    DEVICE_RGB north_west = get_pixel_value_at(pixel, -1, -1, bpp, pitch);
    DEVICE_RGB north = get_pixel_value_at(pixel, -1, 0, bpp, pitch);
    DEVICE_RGB north_east = get_pixel_value_at(pixel, -1, 1, bpp, pitch);
    DEVICE_RGB east = get_pixel_value_at(pixel, 0, 1, bpp, pitch);
    DEVICE_RGB south_east = get_pixel_value_at(pixel, 1, 1, bpp, pitch);
    DEVICE_RGB south = get_pixel_value_at(pixel, 1, 0, bpp, pitch);
    DEVICE_RGB south_west = get_pixel_value_at(pixel, 1, -1, bpp, pitch);
    double verticalx = north_west.r * v_kernel[0][0] + north.r * v_kernel[0][1] + north_east.r * v_kernel[0][2] + west.r * v_kernel[1][0] +
                       center.r * v_kernel[1][1] + east.r * v_kernel[1][2] + south_west.r * v_kernel[2][0] + south.r * v_kernel[2][1] +
                       south_east.r * v_kernel[2][2];
    double verticaly = north_west.g * v_kernel[0][0] + north.g * v_kernel[0][1] + north_east.g * v_kernel[0][2] + west.g * v_kernel[1][0] +
                       center.g * v_kernel[1][1] + east.g * v_kernel[1][2] + south_west.g * v_kernel[2][0] + south.g * v_kernel[2][1] +
                       south_east.g * v_kernel[2][2];
    double verticalz = north_west.b * v_kernel[0][0] + north.b * v_kernel[0][1] + north_east.b * v_kernel[0][2] + west.b * v_kernel[1][0] +
                       center.b * v_kernel[1][1] + east.b * v_kernel[1][2] + south_west.b * v_kernel[2][0] + south.b * v_kernel[2][1] +
                       south_east.b * v_kernel[2][2];
    double horizontalx = north_west.r * h_kernel[0][0] + north.r * h_kernel[0][1] + north_east.r * h_kernel[0][2] + west.r * h_kernel[1][0] +
                         center.r * h_kernel[1][1] + east.r * h_kernel[1][2] + south_west.r * h_kernel[2][0] + south.r * h_kernel[2][1] +
                         south_east.r * h_kernel[2][2];
    double horizontaly = north_west.g * h_kernel[0][0] + north.g * h_kernel[0][1] + north_east.g * h_kernel[0][2] + west.g * h_kernel[1][0] +
                         center.g * h_kernel[1][1] + east.g * h_kernel[1][2] + south_west.g * h_kernel[2][0] + south.g * h_kernel[2][1] +
                         south_east.g * h_kernel[2][2];
    double horizontalz = north_west.b * h_kernel[0][0] + north.b * h_kernel[0][1] + north_east.b * h_kernel[0][2] + west.b * h_kernel[1][0] +
                         center.b * h_kernel[1][1] + east.b * h_kernel[1][2] + south_west.b * h_kernel[2][0] + south.b * h_kernel[2][1] +
                         south_east.b * h_kernel[2][2];
    convolution_directions dir;
    DEVICE_RGB minn = {0, 0, 0, 0};
    DEVICE_RGB maxx = {255, 255, 255, 255};
    uint8_t rh = (uint8_t)normalize(maxx.r, minn.r, horizontalx);
    uint8_t rv = (uint8_t)normalize(maxx.r, minn.r, verticalx);
    uint8_t gh = (uint8_t)normalize(maxx.r, minn.r, horizontaly);
    uint8_t gv = (uint8_t)normalize(maxx.r, minn.r, verticaly);
    uint8_t bh = (uint8_t)normalize(maxx.r, minn.r, horizontalz);
    uint8_t bv = (uint8_t)normalize(maxx.r, minn.r, verticalz);
    DEVICE_RGB vertical = {rv, gv, bv, 0};
    DEVICE_RGB horizontal = {rh, gh, bh, 0};
    dir.vertical = vertical;
    dir.horizontal = horizontal;
    return dir;
  }

  /* pos 0 = vertical convolution kernel
     pos 1 = horizontal convolution kernel */

  AX_DEVICE_ONLY DEVICE_RGB get_convolution_values(uint8_t *pixel, const int bpp, int pitch, uint8_t convolution, uint8_t border) {
    int custom_kernel = 0;
    convolution_directions convoluted;
    if (custom_kernel == 0) {
      if (convolution == AXOMAE_USE_SOBEL)
        convoluted = compute_convolution(pixel, bpp, pitch, sobel_mask_horizontal, sobel_mask_vertical, border);
      else if (convolution == AXOMAE_USE_PREWITT)
        convoluted = compute_convolution(pixel, bpp, pitch, prewitt_mask_horizontal, prewitt_mask_vertical, border);
      else
        convoluted = compute_convolution(pixel, bpp, pitch, scharr_mask_horizontal, scharr_mask_vertical, border);
      DEVICE_RGB var = convoluted.vertical.magnitude_rgb(convoluted.vertical, convoluted.horizontal);
      return var;
    } else {
      // TODO : add custom kernels processing
      return {0, 0, 0, 0};
    }
  }

  AX_DEVICE_ONLY DEVICE_RGB compute_normal(uint8_t *pixel, int bpp, int pitch, double factor) {
    DEVICE_RGB center = get_pixel_value_at(pixel, 0, 0, bpp, pitch);
    DEVICE_RGB west = get_pixel_value_at(pixel, 0, -1, bpp, pitch);
    DEVICE_RGB north_west = get_pixel_value_at(pixel, -1, -1, bpp, pitch);
    DEVICE_RGB north = get_pixel_value_at(pixel, -1, 0, bpp, pitch);
    DEVICE_RGB north_east = get_pixel_value_at(pixel, -1, 1, bpp, pitch);
    DEVICE_RGB east = get_pixel_value_at(pixel, 0, 1, bpp, pitch);
    DEVICE_RGB south_east = get_pixel_value_at(pixel, 1, 1, bpp, pitch);
    DEVICE_RGB south = get_pixel_value_at(pixel, 1, 0, bpp, pitch);
    DEVICE_RGB south_west = get_pixel_value_at(pixel, 1, -1, bpp, pitch);
    float dx = factor * (east.g - west.g) / 255;
    float dy = factor * (north.g - south.g) / 255;
    float ddx = factor * (north_east.g - south_west.g) / 255;
    float ddy = factor * (north_west.g - south_east.g) / 255;
    float Nx = normalize(-1, 1, lerp(dy, ddy, 0.5));
    float Ny = normalize(-1, 1, lerp(dx, ddx, 0.5));
    uint8_t Nz = 255;
    if (Nx >= 255)
      Nx = 255;
    else if (Nx <= 0)
      Nx = 0;
    if (Ny >= 255)
      Ny = 255;
    else if (Ny <= 0)
      Ny = 0;
    return {(uint8_t)std::floor(Nx), (uint8_t)std::floor(Ny), Nz, 0};
  }

  AX_DEVICE_ONLY static void replace_min(DEVICE_RGB rgb) {
    uint32_t *max = &max_int_rgb;
    uint8_t *pixel = (uint8_t *)max;
    DEVICE_RGB maxx = int_to_rgb(pixel, 4);
    maxx.r = maxx.r >= rgb.r ? maxx.r : rgb.r;
    maxx.g = maxx.g >= rgb.g ? maxx.g : rgb.g;
    maxx.b = maxx.b >= rgb.b ? maxx.b : rgb.b;
    *max = rgb_to_int(maxx);
  }

  AX_DEVICE_ONLY static void replace_max(DEVICE_RGB rgb) {
    uint32_t *min = &min_int_rgb;
    uint8_t *pixel = (uint8_t *)min;
    DEVICE_RGB minn = int_to_rgb(pixel, 4);
    minn.r = minn.r < rgb.r ? minn.r : rgb.r;
    minn.g = minn.g < rgb.g ? minn.g : rgb.g;
    minn.b = minn.b < rgb.b ? minn.b : rgb.b;
    *min = rgb_to_int(minn);
  }

  /* kernels */
  /*********************************************************************************************************************************************/
  __global__ void GPU_compute_greyscale(void *array, int size_w, int size_h, const int bpp, int pitch, const bool luminance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size_w && j < size_h) {
      DEVICE_RGB rgb = {0, 0, 0, 0};
      uint8_t *pixel_value = (uint8_t *)(array) + i * bpp + j * pitch;
      rgb = compute_greyscale(int_to_rgb(pixel_value, bpp), luminance);
      set_pixel_color(pixel_value, rgb, bpp);
    }
  }
  __global__ void GPU_compute_edges(
      void *image, void *save, unsigned int width, unsigned int height, int bpp, int pitch, uint8_t convolution, uint8_t border) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < width - 1 && j < height - 1 && i > 0 && j > 0) {
      uint8_t *pixel = (uint8_t *)(image) + i * bpp + j * pitch;
      uint8_t *p = (uint8_t *)(save) + i * bpp + j * pitch;
      DEVICE_RGB rgb = get_convolution_values(pixel, bpp, pitch, convolution, border);
      set_pixel_color(p, rgb, bpp);
    } else {  // just some weird cases , TODO later
      uint8_t *pixel = (uint8_t *)(image) + i * bpp + j * pitch;
      if (border == AXOMAE_REPEAT) {
      } else if (border == AXOMAE_CLAMP) {
      } else {
      }
    }
  }

  __global__ void GPU_compute_normals(
      void *image, void *save, unsigned int width, unsigned int height, int bpp, int pitch, double factor, uint8_t border) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    uint8_t *pixel = (uint8_t *)(image) + i * bpp + j * pitch;
    uint8_t *write = (uint8_t *)(save) + i * bpp + j * pitch;
    if (i < width - 1 && j < height - 1 && i > 1 && j > 1) {
      DEVICE_RGB rgb = compute_normal(pixel, bpp, pitch, factor);
      set_pixel_color(write, rgb, bpp);
    } else {
      if (border == AXOMAE_REPEAT) {
      } else if (border == AXOMAE_CLAMP) {
      } else {
      }
    }
  }

  /*host functions*/
  /*********************************************************************************************************************************************/

  void GPU_compute_greyscale(SDL_Surface *image, const bool luminance) {
    int width = image->w;
    int height = image->h;
    int pitch = image->pitch;
    int bpp = image->format->BytesPerPixel;
    void *D_image;
    size_t size = pitch * height;
    cudaMalloc((void **)&D_image, size);
    cudaMemcpy(D_image, image->pixels, size, cudaMemcpyHostToDevice);
    gpu_threads D = get_optimal_thread_distribution(width, height);
    GPU_compute_greyscale<<<D.blocks, D.threads>>>(D_image, width, height, bpp, pitch, luminance);
    check_error(__FILE__, __LINE__);
    SDL_LockSurface(image);
    cudaMemcpy(image->pixels, D_image, size, cudaMemcpyDeviceToHost);
    SDL_UnlockSurface(image);
    cudaFree(D_image);
  }

  void GPU_compute_height(SDL_Surface *greyscale, uint8_t convolution, uint8_t border) {
    SDLSurfParam param(greyscale);
    void *D_image, *R_image;
    size_t size = param.getByteSize();
    cudaMalloc((void **)&D_image, size);
    cudaMalloc((void **)&R_image, size);
    cudaMemcpy(D_image, param.data, size, cudaMemcpyHostToDevice);
    gpu_threads D = get_optimal_thread_distribution(param.width, param.height);
    D.blocks.x++;  // border management
    D.blocks.y++;  //
    GPU_compute_edges<<<D.blocks, D.threads>>>(D_image, R_image, param.width, param.height, param.bpp, param.pitch, convolution, border);
    check_error(__FILE__, __LINE__);
    SDL_LockSurface(greyscale);
    cudaMemcpy(greyscale->pixels, R_image, size, cudaMemcpyDeviceToHost);
    SDL_UnlockSurface(greyscale);
    cudaFree(D_image);
    cudaFree(R_image);
  }

  void GPU_compute_normal(SDL_Surface *height, double factor, uint8_t border) {

    SDLSurfParam param(height);
    void *D_image, *D_save;
    cudaMalloc((void **)&D_image, param.getByteSize());
    cudaMalloc((void **)&D_save, param.getByteSize());

    cudaMemcpy(D_image, param.data, param.getByteSize(), cudaMemcpyHostToDevice);
    gpu_threads blocks = get_optimal_thread_distribution(param.width, param.height);
    blocks.blocks.x++;
    blocks.blocks.y++;
    GPU_compute_normals<<<blocks.blocks, blocks.threads>>>(D_image, D_save, param.width, param.height, param.bpp, param.pitch, factor, border);
    check_error(__FILE__, __LINE__);
    SDL_LockSurface(height);
    cudaMemcpy(height->pixels, D_save, param.getByteSize(), cudaMemcpyDeviceToHost);
    SDL_UnlockSurface(height);

    cudaFree(D_image);
    cudaFree(D_save);
  }

};  // namespace axomae
