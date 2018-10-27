#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <cstdint>
#include <iostream>
#include <future>
#include <cmath>
#include <vector>
#include <SDL2/SDL.h> 
namespace axomae {

	
	

	
	
	
	struct Object3D{
		std::vector<float> vertices; 
		std::vector<float> uv ; 
		std::vector<float> normals; 
		std::vector<float> bitangents ; 
		std::vector<float> tangents;
		std::vector<unsigned int> indices;

	
	};
	
	
	struct Point2D{

		float x ; 
		float y ; 

		void print(){
			std::cout << x << "     " << y << "\n" ; 
		}
	};
	
	struct Vect3D {

		float x ; 
		float y ; 
		float z ; 
	
		void print(){
			std::cout << x << "     " << y << "      " << z <<  "\n" ; 
		}
		

		auto magnitude() {
		
			return sqrt(x*x + y*y + z*z); 

		}

		void normalize(){
			auto mag = this->magnitude() ; 
			x = abs(x/mag) ; 
			y = abs(y/mag) ; 
			z = abs(z/mag) ; 

		}
		auto dot(Vect3D V ) {
			return x * V.x + y * V.y + z * V.z ; 

		}
		

	};
	
	
	
	
	
	
	
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
