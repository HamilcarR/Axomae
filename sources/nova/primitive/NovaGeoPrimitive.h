#ifndef NOVAGEOPRIMITIVE_H
#define NOVAGEOPRIMITIVE_H
#include "nova_primitive.h"

namespace material {
  class NovaMaterialInterface;
}

namespace shape {
  class NovaShapeInterface;
}

namespace nova::primitive {
  class NovaGeoPrimitive final : public NovaPrimitiveInterface {
   private:
    std::unique_ptr<material::NovaMaterialInterface> material{};
    std::unique_ptr<shape::NovaShapeInterface> shape{};

   public:
    NovaGeoPrimitive() = default;
    NovaGeoPrimitive(std::unique_ptr<shape::NovaShapeInterface> &shape, std::unique_ptr<material::NovaMaterialInterface> &material);
    ~NovaGeoPrimitive() override;
    NovaGeoPrimitive(const NovaGeoPrimitive &other) = delete;
    NovaGeoPrimitive(NovaGeoPrimitive &&other) noexcept = default;
    NovaGeoPrimitive &operator=(const NovaGeoPrimitive &other) = delete;
    NovaGeoPrimitive &operator=(NovaGeoPrimitive &&other) noexcept = default;

    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const override;
    bool scatter(const Ray &in, Ray &out, hit_data &data) const override;
    [[nodiscard]] glm::vec3 centroid() const override;

    [[nodiscard]] geometry::BoundingBox computeAABB() const override;
  };
}  // namespace nova::primitive
#endif  // NOVAGEOPRIMITIVE_H
