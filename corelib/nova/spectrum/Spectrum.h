#ifndef SPECTRUM_H
#define SPECTRUM_H
#include "spectrum/BaseSpectrum.h"
namespace nova {

  // we ignore the wave behavior of light until the engine is numerically stable.

  class SampledWavelength {};
  class RgbColorSpace {};

  class WaveSpectrum : public BaseSpectrum<WaveSpectrum> {
    static constexpr unsigned SPECTRUM_SAMPLES = 4;
    float samples[SPECTRUM_SAMPLES]{};
  };

  class ColorSpectrum : public BaseSpectrum<ColorSpectrum> {
    static constexpr unsigned SPECTRUM_SAMPLES = 3;
    float samples[SPECTRUM_SAMPLES]{};

   public:
    ax_device_callable_inlined ColorSpectrum() = default;

    ax_device_callable_inlined ColorSpectrum(float v) {
      for (unsigned i = 0; i < SPECTRUM_SAMPLES; i++)
        samples[i] = v;
    }

    ax_device_callable_inlined glm::vec3 toRgb(const SampledWavelength &wl, const RgbColorSpace &cs) {
      return glm::vec3(samples[0], samples[1], samples[2]);
    }

    ax_device_callable_inlined float *getSamplesArray() { return samples; }
    ax_device_callable_inlined const float *getSamplesArray() const { return samples; }
    ax_device_callable_inlined unsigned getSamplesSize() const { return SPECTRUM_SAMPLES; }
  };

  using Spectrum = std::conditional_t<core::build::is_spectrum_build, WaveSpectrum, ColorSpectrum>;

}  // namespace nova
#endif  // SPECTRUM_H
