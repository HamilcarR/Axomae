#ifndef UTILS_3D_H
#define UTILS_3D_H




#include "constants.h" 



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

}

#endif
