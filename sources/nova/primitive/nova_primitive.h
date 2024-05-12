#ifndef NOVA_PRIMITIVE_H
#define NOVA_PRIMITIVE_H
#include "Axomae_macros.h"
#include "Hitable.h"
#include <memory>
namespace nova::shape {
  class NovaShapeInterface;
}
namespace nova::material {
  class NovaMaterialInterface;
}
namespace nova::primitive {
  class NovaPrimitiveInterface : public Hitable {
   public:
    ~NovaPrimitiveInterface() override = default;
    virtual bool scatter(const Ray &in, Ray &out, hit_data &data) const = 0;

    template<class SUBCLASS, class... Args>
    static std::unique_ptr<SUBCLASS> create(Args &&...args) {
      ASSERT_SUBTYPE(NovaPrimitiveInterface, SUBCLASS);
      return std::make_unique<SUBCLASS>(std::forward<Args>(args)...);
    }
  };

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

    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const base_options *user_options) const override;
    bool scatter(const Ray &in, Ray &out, hit_data &data) const override;
  };
}  // namespace nova::primitive
#endif  // NOVA_PRIMITIVE_H
