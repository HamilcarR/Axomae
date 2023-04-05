#include "../includes/Renderer.h"


using namespace axomae ; 


Renderer::Renderer(){
	start_draw = false ;
	texture_database = TextureDatabase::getInstance(); 
	mouse_state.pos_x = 0 ;  
	mouse_state.pos_y = 0 ; 
	mouse_state.left_button_clicked = false ;
	mouse_state.left_button_released = true ; 
	mouse_state.right_button_clicked = false ;
	mouse_state.right_button_released = true ; 
	mouse_state.previous_pos_x = 0 ; 
	mouse_state.previous_pos_y = 0 ;  
	scene_camera = new ArcballCamera(45.f , &screen_size ,  0.001f , 1000.f , 100.f, &mouse_state); 	
}

Renderer::~Renderer(){
	for(unsigned int i = 0 ; i < scene.size() ; i++){
		if(scene[i] != nullptr){
			scene[i]->clean();
			delete scene[i]; 
		}
	}
	scene.clear(); 
	texture_database->clean();
	delete scene_camera ; 
}

void Renderer::initialize(){
	glEnable(GL_DEPTH_TEST); 
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
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
		for(Drawable *A : scene){
			A->start_draw(); 
			A->setSceneCameraPointer(scene_camera); 
		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
		return true; 				
	}
	else{

		glClearColor(0 , 0 , 0, 1.f);
		return false ;	
	}
}

void Renderer::draw(QOpenGLFunctions_4_3_Core* gl){
	scene_camera->computeViewSpace(); 
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
		scene.push_back(new Drawable(m)); 
	start_draw = true ;
}

void Renderer::setScreenSize(unsigned int width , unsigned int height){
	screen_size.width = width; 
	screen_size.height = height; 
}

void Renderer::onLeftClick(){
	scene_camera->onLeftClick(); 
}
void Renderer::onRightClick(){
	scene_camera->onRightClick(); 
}
void Renderer::onLeftClickRelease(){
	scene_camera->onLeftClickRelease(); 
}
void Renderer::onRightClickRelease(){
	scene_camera->onRightClickRelease();
}
	




