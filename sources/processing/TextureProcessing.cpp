#include "TextureProcessing.h"
#include "GenericTextureProcessing.h"
#include "cuda/CubemapProcessing.cuh"

namespace gpu_func = gpgpu_functions::irradiance_mapping;
template<>
std::unique_ptr<TextureData> TextureOperations<float>::computeDiffuseIrradiance(unsigned _width, unsigned _height, unsigned delta, bool gpu) const {
  if (!isDimPowerOfTwo(_width) || !isDimPowerOfTwo(_height))
    throw TextureNonPowerOfTwoDimensionsException();
  TextureData envmap_tex_data;
  envmap_tex_data.width = _width;
  envmap_tex_data.height = _height;
  envmap_tex_data.mipmaps = 0;
  if (gpu) {
    envmap_tex_data.f_data.resize(_width * _height * 4);
    envmap_tex_data.nb_components = 4;
    std::vector<float> temp;
    temp.reserve(data->size());
    for (int i = 0; i < data->size(); i += 3) {
      temp.push_back((*data)[i]);
      temp.push_back((*data)[i + 1]);
      temp.push_back((*data)[i + 2]);
      temp.push_back(0);  // Alpha channel because cuda channel descriptors don't have RGB only
    }
    float *dest_array{};
    envmap_tex_data.nb_components = 4;
    gpu_func::GPU_compute_irradiance(
        temp.data(), width, height, envmap_tex_data.nb_components, &dest_array, envmap_tex_data.width, envmap_tex_data.height, delta);
    for (unsigned i = 0; i < envmap_tex_data.f_data.size(); i++)
      envmap_tex_data.f_data[i] = dest_array[i];
  } else {
    envmap_tex_data.f_data.resize(_width * _height * channels);
    envmap_tex_data.nb_components = channels;
    std::vector<std::shared_future<void>> futures;
    for (unsigned i = 1; i <= MAX_THREADS; i++) {
      unsigned int width_max = (_width / MAX_THREADS) * i, width_min = width_max - (_width / MAX_THREADS);
      if (i == MAX_THREADS)
        width_max += _width % MAX_THREADS - 1;
      auto lambda = [this, &envmap_tex_data](unsigned delta, const unsigned width_min, const unsigned width_max) {
        this->launchAsyncDiffuseIrradianceCompute(
            delta, envmap_tex_data.f_data.data(), width_min, width_max, envmap_tex_data.width, envmap_tex_data.height);
      };
      futures.push_back(std::async(std::launch::async, lambda, delta, width_min, width_max));
    }
    for (const auto &it : futures) {
      it.get();
    }
  }
  return std::make_unique<TextureData>(envmap_tex_data);
}
