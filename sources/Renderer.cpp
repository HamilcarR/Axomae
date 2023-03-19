#include "../includes/Renderer.h"


using namespace axomae ; 


Renderer::Renderer(){
	start_draw = false ;	
}

Renderer::~Renderer(){
	for(unsigned int i = 0 ; i < scene.size() ; i++){
		if(scene[i] != nullptr){
			scene[i]->clean();
			delete scene[i]; 
		}
	}
	scene.clear(); 
}

void Renderer::initialize(){
	glEnable(GL_DEPTH_TEST);
}


bool Renderer::scene_ready(){
	for(Drawable *object : scene)
		if(!object->ready())
			return false;
	return true ; 
}

bool Renderer::prep_draw(){
	if(start_draw && scene_ready()){
		/*Bind buffers*/
		for(Drawable *A : scene)
			A->start_draw(); 	
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
		return true; 				
	}
	else{
		glClearColor(0 , 0 , 0, 1.f);
		return false ;	
	}
}

void Renderer::draw(QOpenGLFunctions_4_3_Core* gl){
	for (Drawable *A : scene){
		A->bind(); 
		gl->glDrawElements(GL_TRIANGLES , A->mesh_object->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		A->unbind();
	}
}



void Renderer::end_draw(){
	for(Drawable *A : scene)
		A->end_draw(); 
	
}

void Renderer::set_new_scene(std::vector<Mesh> &new_scene){
	for (Drawable *A : scene){
		A->clean(); 
		delete A ; 
	}
	scene.clear(); 
	for (Mesh m : new_scene)
		scene.push_back(new Drawable(&m)); 
	start_draw = true ;
}


