#include "TextureProcessing.h"
#include "GenericTextureProcessing.h"

#include <internal/debug/Logger.h>
#if defined(AXOMAE_USE_CUDA)
#  include "cuda/CubemapProcessing.h"
#endif

template<>
image::ImageHolder<float> TextureOperations<float>::computeDiffuseIrradiance(unsigned _width, unsigned _height, unsigned delta, bool gpu) const {
  if (!isDimPowerOfTwo(_width) || !isDimPowerOfTwo(_height))
    throw TextureNonPowerOfTwoDimensionsException();
  image::ImageHolder<float> envmap_tex_data;
  envmap_tex_data.metadata.width = _width;
  envmap_tex_data.metadata.height = _height;
  envmap_tex_data.data().resize(_width * _height * channels);
  envmap_tex_data.metadata.channels = 4;

  if (gpu) {
#if defined(AXOMAE_USE_CUDA)
    namespace gpu_func = gpgpu_functions::irradiance_mapping;
    std::vector<float> temp;
    temp.resize(width * height * 4);
    for (int i = 0; i < width * height; i++) {
      int idx = i * channels;
      int t_idx = i * 4;
      temp[t_idx] = data[idx];
      temp[t_idx + 1] = data[idx + 1];
      temp[t_idx + 2] = data[idx + 2];
      temp[t_idx + 3] = 1.f;
    }
    float *dest_array{};

    gpu_func::GPU_compute_irradiance(temp.data(),
                                     width,
                                     height,
                                     envmap_tex_data.metadata.channels,
                                     &dest_array,  // TODO : replace with managed vector ?
                                     envmap_tex_data.metadata.width,
                                     envmap_tex_data.metadata.height,
                                     delta);
    for (unsigned i = 0; i < envmap_tex_data.data().size(); i++)
      envmap_tex_data.data()[i] = dest_array[i];
#else
    LOG("CUDA not found. Enable 'AXOMAE_USE_CUDA' in build if this platform has an Nvidia GPU.", LogLevel::ERROR);
#endif
  } else {
    std::vector<std::shared_future<void>> futures;
    for (unsigned i = 1; i <= MAX_THREADS; i++) {
      unsigned int width_max = (_width / MAX_THREADS) * i, width_min = width_max - (_width / MAX_THREADS);
      if (i == MAX_THREADS)
        width_max += _width % MAX_THREADS - 1;
      auto lambda = [this, &envmap_tex_data](unsigned delta, const unsigned width_min, const unsigned width_max) {
        this->launchAsyncDiffuseIrradianceCompute(
            delta, envmap_tex_data.data().data(), width_min, width_max, envmap_tex_data.metadata.width, envmap_tex_data.metadata.height);
      };
      futures.push_back(std::async(std::launch::async, lambda, delta, width_min, width_max));
    }
    for (const auto &it : futures) {
      it.get();
    }
  }
  return envmap_tex_data;
}
