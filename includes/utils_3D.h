#ifndef UTILS_3D_H
#define UTILS_3D_H

#include "constants.h" 


/*TODO : Create classes , implement clean up */
namespace axomae {


struct Object3D{
	std::vector<float> vertices; 
	std::vector<float> uv ; 
	std::vector<float> normals; 
	std::vector<float> bitangents ; 
	std::vector<float> tangents;
	std::vector<unsigned int> indices;
};

struct TextureData{
	unsigned int width ; 
	unsigned int height ; 
	std::string name ; 
	uint32_t *data ; 
		

	TextureData(){
		width = 0 ; 
		height = 0 ; 
		data = nullptr ; 
	}	
	~TextureData(){}
	
	/*provides deep copy of the object , but doesn't do the cleanup for the copied object*/
	TextureData& operator=(const TextureData& from){ 
		width = from.width ;
		height = from.height ; 
		data = new uint32_t [from.width * from.height] ; 
		memcpy((void*) data , (void*) from.data , from.width * from.height * sizeof(uint32_t));		
		name = from.name ; 
		return *this ; 
	}

	void clean(){
		if(data != nullptr)
			delete data ; 
		data = nullptr ;
		width = 0 ; 
		height = 0 ;
		name = "" ; 

	}
};

struct TexturePack{
	TextureData diffuse ; 
	TextureData normal ; 
	TextureData opacity ; 
	TextureData metallic ; 
	TextureData roughness ;
	TextureData optional ; 
}; 

//For PBR workflow
struct Material{
	TexturePack textures ;
	float dielectric_factor; //metallic factor , 0.0 = full dielectric , 1.0 = full metallic
	float roughtness_factor; //0.0 = smooth , 1.0 = rough 	
	float transmission_factor; //defines amount of light transmitted through the surface
	float emissive_factor;
};

//make it proper class
struct Mesh{
	Object3D geometry ; 
	Material material ;
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
