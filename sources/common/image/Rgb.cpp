#include "Rgb.h"
#include "constants.h"
#include <sstream>

namespace image {

  Rgb::Rgb() : red(0.), green(0.), blue(0.), alpha(0.) {}

  Rgb::Rgb(float r, float g, float b) : red(r), green(g), blue(b), alpha(0) {}

  Rgb::Rgb(float r, float g, float b, float a) : red(r), green(g), blue(b), alpha(a) {}

  std::string Rgb::to_string() const {
    std::stringstream os;

    os << "R: " << std::to_string(red) << " ";
    os << "G: " << std::to_string(green) << " ";
    os << "B: " << std::to_string(blue) << " ";
    os << "A: " << std::to_string(alpha) << "  ";
    return os.str();
  }

  [[maybe_unused]] double Rgb::intensity() {
    double av = (red + green + blue) / 3;
    return av / 255;
  }

  void Rgb::clamp() {
    if (red > 255)
      red = 255;
    if (green > 255)
      green = 255;
    if (blue > 255)
      blue = 255;
    if (red < 0)
      red = 0;
    if (green < 0)
      green = 0;
    if (blue < 0)
      blue = 0;
  }

  [[maybe_unused]] Rgb Rgb::int_to_rgb(uint32_t val) {
    Rgb rgb;
    rgb.alpha = static_cast<float>(val >> 24 & 0XFF);
    rgb.blue = static_cast<float>(val >> 16 & 0XFF);
    rgb.green = static_cast<float>(val >> 8 & 0XFF);
    rgb.red = static_cast<float>(val & 0XFF);
    return rgb;
  }

  [[maybe_unused]] Rgb Rgb::int_to_rgb(uint16_t val) {
    Rgb rgb;
    rgb.alpha = static_cast<float>(val >> 12 & 0XF);
    rgb.blue = static_cast<float>(val >> 8 & 0XF);
    rgb.green = static_cast<float>(val >> 4 & 0XF);
    rgb.red = static_cast<float>(val & 0XF);
    return rgb;
  }

  uint32_t Rgb::rgb_to_int() {
    uint32_t image = 0;
    int b = static_cast<int>(std::round(blue));
    int g = static_cast<int>(std::round(green));
    int r = static_cast<int>(std::round(red));
    int a = static_cast<int>(std::round(alpha));
    image = r | (g << 8) | (b << 16) | (a << 24);
    return image;
  }

  bool Rgb::operator==(const Rgb &rgb) { return red == rgb.red && green == rgb.green && blue == rgb.blue && alpha == rgb.alpha; }

  template<typename T>
  Rgb Rgb::operator*(T arg) const {
    Rgb rgb = Rgb(static_cast<float>(red * arg), static_cast<float>(green * arg), static_cast<float>(blue * arg), static_cast<float>(alpha * arg));
    return rgb;
  }

  Rgb Rgb::operator+=(float arg) const {
    Rgb rgb = Rgb(red + arg, green + arg, blue + arg, alpha + arg);
    return rgb;
  }

  Rgb Rgb::operator+=(Rgb arg) const {
    Rgb rgb = Rgb(red + arg.red, green + arg.green, blue + arg.blue, alpha + arg.alpha);
    return rgb;
  }

  Rgb Rgb::operator+(Rgb arg) const {
    Rgb rgb = Rgb(red + arg.red, green + arg.green, blue + arg.blue, alpha + arg.alpha);
    return rgb;
  }

  Rgb Rgb::operator/(float arg) const {
    assert(arg > 0);
    Rgb rgb = Rgb(red / arg, green / arg, blue / arg, alpha / arg);
    return rgb;
  }

  Rgb Rgb::operator-(Rgb arg) const {
    Rgb rgb = Rgb(red - arg.red, green - arg.green, blue - arg.blue);
    return rgb;
  }

  [[maybe_unused]] void Rgb::invert_color() {
    red = std::abs(red - 255);
    green = std::abs(green - 255);
    blue = std::abs(blue - 255);
    alpha = std::abs(alpha - 255);
  }
}  // namespace image