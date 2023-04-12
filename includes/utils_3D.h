#ifndef UTILS_3D_H
#define UTILS_3D_H

#include "constants.h" 
#include "DebugGL.h" 
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp> 







inline void errorCheck(){
	GLenum error = GL_NO_ERROR;
    	error = glGetError();
    	if (error != GL_NO_ERROR) 
        	std::cout << "Error:" << error << std::endl ; 

}







/*TODO : Create classes , implement clean up */
namespace axomae {


struct Object3D{
	std::vector<float> vertices; 
	std::vector<float> uv ; 
	std::vector<float> colors; 
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
	friend std::ostream& operator<<(std::ostream& os , const Point2D& p) {
		os << "(" << p.x << "," << p.y << ")" ; 
		return os ; 	
	}
};	


struct Vect3D {
	float x ; 
	float y ; 
	float z ;

	friend std::ostream& operator<<(std::ostream& os , const Vect3D& v){
		os << "(" << v.x << "," << v.y << "," << v.z << ")" ; 
		return os ; 
	}

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
