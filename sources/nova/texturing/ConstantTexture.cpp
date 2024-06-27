#include "ConstantTexture.h"
using namespace nova::texturing;

ConstantTexture::ConstantTexture(const glm::vec4 &albedo_) : albedo(albedo_) {}

glm::vec4 ConstantTexture::sample(float u, float v, const texture_sample_data &sample_data) const { return albedo; }
