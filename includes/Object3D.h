#ifndef OBJECT3D_H
#define OBJECT3D_H


#include "constants.h" 





class Object3D{

public:
	std::vector<float> vertices;	/*<Vertices array*/ 
	std::vector<float> uv ;			/*<UV arrays of dimension 2*/ 
	std::vector<float> colors; 		/*<Colors array , Format is RGB*/
	std::vector<float> normals; 	/*<Normals of the geometry*/
	std::vector<float> bitangents ; /*<Bitangent of each vertex*/
	std::vector<float> tangents;	/*<Tangent of each vertex*/
	std::vector<unsigned int> indices;	/*<Indices of the vertices buffer*/
};


















#endif