#ifndef VECTOR_H
#define VECTOR_H
#include <cmath>
#include <ostream>
namespace math::geometry {
  class IVector {
   public:
    [[nodiscard]] virtual float magnitude() const = 0;
    [[nodiscard]] virtual float dot(const IVector &arg) const = 0;
    virtual void normalize() = 0;
  };

  class Vect2D : public IVector {
   public:
    Vect2D() = default;
    Vect2D(float x, float y);
    [[nodiscard]] float magnitude() const override;
    void normalize() override;
    [[nodiscard]] float dot(const IVector &arg) const override;
    friend std::ostream &operator<<(std::ostream &os, const Vect2D &p);

   public:
    float x{};
    float y{};
  };

  class Vect3D : public IVector {
   public:
    Vect3D() = default;
    Vect3D(float x, float y, float z);
    [[nodiscard]] float magnitude() const override;
    void normalize() override;
    [[nodiscard]] float dot(const IVector &arg) const override;
    friend std::ostream &operator<<(std::ostream &os, const Vect3D &v);

   public:
    float x{};
    float y{};
    float z{};
  };

}  // namespace math::geometry

#endif  // VECTOR_H
