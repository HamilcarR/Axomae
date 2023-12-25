#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <future>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

namespace axomae {

  constexpr int INT_MAXX = 30000;
  constexpr uint8_t AXOMAE_USE_SOBEL = 0X00;
  constexpr uint8_t AXOMAE_USE_PREWITT = 0X01;
  constexpr uint8_t AXOMAE_USE_SCHARR = 0X02;
  constexpr uint8_t AXOMAE_CLAMP = 0XFF;
  constexpr uint8_t AXOMAE_REPEAT = 0X00;
  constexpr uint8_t AXOMAE_MIRROR = 0X01;
  constexpr uint8_t AXOMAE_GHOST = 0X02;
  constexpr uint8_t AXOMAE_RED = 0X00;
  constexpr uint8_t AXOMAE_GREEN = 0X01;
  constexpr uint8_t AXOMAE_BLUE = 0X02;
  constexpr uint8_t KERNEL_SIZE = 3;
  constexpr int SOBEL = 2;
  constexpr int PREWITT = 1;

}  // namespace axomae

struct MouseState {
  unsigned int pos_x;
  unsigned int pos_y;
  bool left_button_clicked;
  bool left_button_released;
  bool right_button_clicked;
  bool right_button_released;
  unsigned int previous_pos_x;
  unsigned int previous_pos_y;
  bool busy;
};

struct ScreenSize {
  unsigned int width;
  unsigned int height;
};

#endif
