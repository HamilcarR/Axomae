#ifndef NOVAGEOPRIMITIVE_H
#define NOVAGEOPRIMITIVE_H
#include "nova_primitive.h"

namespace nova::primitive {
  class NovaGeoPrimitive final : public NovaPrimitiveInterface {
   private:
    material::NovaMaterialInterface *material{};
    shape::NovaShapeInterface *shape{};

   public:
    NovaGeoPrimitive() = default;
    NovaGeoPrimitive(shape::NovaShapeInterface *shape, material::NovaMaterialInterface *material);
    ~NovaGeoPrimitive() override = default;
    NovaGeoPrimitive(const NovaGeoPrimitive &other) = default;
    NovaGeoPrimitive(NovaGeoPrimitive &&other) noexcept = default;
    NovaGeoPrimitive &operator=(const NovaGeoPrimitive &other) = default;
    NovaGeoPrimitive &operator=(NovaGeoPrimitive &&other) noexcept = default;

    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const override;
    bool scatter(const Ray &in, Ray &out, hit_data &data) const override;
    [[nodiscard]] glm::vec3 centroid() const override;

    [[nodiscard]] geometry::BoundingBox computeAABB() const override;
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H
