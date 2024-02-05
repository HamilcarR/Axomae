#ifndef RGB_H
#define RGB_H
#include "Axomae_macros.h"
#include "constants.h"

namespace image {
  class Rgb {
   public:
    Rgb();
    Rgb(float r, float g, float b, float a);
    Rgb(float r, float g, float b);
    Rgb(const Rgb &copy);
    Rgb(Rgb &&move) noexcept;
    Rgb &operator=(const Rgb &copy);
    Rgb &operator=(Rgb &&move) noexcept;
    ~Rgb() = default;
    static Rgb int_to_rgb(uint32_t value);
    static Rgb int_to_rgb(uint16_t value);
    double intensity();
    void invert_color();
    bool operator==(const Rgb &arg);
    template<typename T>
    Rgb operator*(T arg) const;
    Rgb operator+=(float arg) const;
    Rgb operator+=(Rgb arg) const;
    Rgb operator+(Rgb arg) const;
    Rgb operator/(float arg) const;
    Rgb operator-(Rgb arg) const;
    void clamp();
    uint32_t rgb_to_int();
    void to_string();
    float red;
    float green;
    float blue;
    float alpha;
  };
}  // namespace image
#endif  // RGB_H
