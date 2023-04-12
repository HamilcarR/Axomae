#ifndef RENDERER_H
#define RENDERER_H

#include "constants.h"
#include "utils_3D.h"
#include "Loader.h"
#include "Mesh.h"
#include "Drawable.h" 
#include "TextureDatabase.h" 
#include "Camera.h"


class Renderer {
public:
	Renderer();
	virtual ~Renderer();	
	void initialize();
	bool prep_draw();
	void draw(); 
	void set_new_scene(std::vector<axomae::Mesh> &new_scene);
	bool scene_ready() ; 
	MouseState* getMouseStatePointer(){ return &mouse_state;} ;  
	void onLeftClick(); 
	void onRightClick(); 
	void onLeftClickRelease(); 
	void onRightClickRelease();
	void onScrollDown(); 
	void onScrollUp(); 
	void setScreenSize(unsigned int width , unsigned int height);

public:
	std::vector<Drawable*> scene ; 
	bool start_draw ; 
	TextureDatabase* texture_database; 		
	Camera *scene_camera ;
	MouseState mouse_state ;
	ScreenSize screen_size ; 	
};

#endif
