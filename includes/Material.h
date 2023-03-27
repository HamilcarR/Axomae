#ifndef MATERIAL_H
#define MATERIAL_H

#include "TextureGroup.h" 
#include "Shader.h" 



class Material{
public:
	Material(); 
	virtual ~Material(); 
	virtual void addTexture(TextureData &data , Texture::TYPE type);
	virtual void bind();
	virtual void clean();
	virtual void initializeMaterial(); 
private:
	TextureGroup textures_group ;
	float dielectric_factor; //metallic factor , 0.0 = full dielectric , 1.0 = full metallic
	float roughtness_factor; //0.0 = smooth , 1.0 = rough 	
	float transmission_factor; //defines amount of light transmitted through the surface
	float emissive_factor;


};







#endif
