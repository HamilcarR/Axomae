#ifndef RENDERER_H
#define RENDERER_H

#include "constants.h"
#include "utils_3D.h"
#include "Loader.h"
#include "Mesh.h"
#include "Drawable.h" 
#include "TextureDatabase.h" 



class Renderer {
public:
	Renderer();
	virtual ~Renderer();	
	void initialize();
	bool prep_draw();
	void draw(QOpenGLFunctions_4_3_Core* api_functions); 
	void end_draw();
	void set_new_scene(std::vector<axomae::Mesh> &new_scene);
	bool scene_ready() ; 
public:
	std::vector<Drawable*> scene ; 
	bool start_draw ; 
	TextureDatabase* texture_database; 		
};

#endif
