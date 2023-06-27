#ifndef OBJECT3D_H
#define OBJECT3D_H


#include "constants.h" 





class Object3D{

public:
	std::vector<float> vertices; 
	std::vector<float> uv ; 
	std::vector<float> colors; 
	std::vector<float> normals; 
	std::vector<float> bitangents ; 
	std::vector<float> tangents;
	std::vector<unsigned int> indices;
};


















#endif