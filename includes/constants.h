#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <cstdint>


namespace axomae {
	const int INT_MAXX = 30000;
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
	static constexpr int SOBEL = 2;
	static constexpr int PREWITT = 1;

























};

















#endif