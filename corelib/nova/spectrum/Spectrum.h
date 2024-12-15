#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <internal/memory/tag_ptr.h>

class RGBSpectrum {
 private:
  float r, g, b;

 public:
  RGBSpectrum(float r, float g, float b);
};

class WaveSpectrum {};

class SpectrumInterface : core::tag_ptr<RGBSpectrum, WaveSpectrum> {};

#endif  // SPECTRUM_H
