#ifndef RGB_H
#define RGB_H
#include <cinttypes>
#include <internal/macro/project_macros.h>
#include <string>

namespace image {
  class Rgb {
   public:
    float red;
    float green;
    float blue;
    float alpha;

   public:
    Rgb();
    Rgb(float r, float g, float b, float a);
    Rgb(float r, float g, float b);
    Rgb(const Rgb &copy) = default;
    Rgb(Rgb &&move) noexcept = default;
    Rgb &operator=(const Rgb &copy) = default;
    Rgb &operator=(Rgb &&move) noexcept = default;
    ~Rgb() = default;
    [[maybe_unused]] static Rgb int_to_rgb(uint32_t value);
    [[maybe_unused]] static Rgb int_to_rgb(uint16_t value);
    [[maybe_unused]] double intensity();
    [[maybe_unused]] void invert_color();
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
    ax_no_discard std::string to_string() const;
    ax_no_discard std::string to_stringi() const;
  };
}  // namespace image
#endif  // RGB_H
