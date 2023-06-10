#include "../includes/Renderer.h"
#include "../includes/Loader.h"

using namespace axomae ; 


Renderer::Renderer(){
	start_draw = false ;	
	camera_framebuffer = nullptr;  
	mouse_state.pos_x = 0 ;  
	mouse_state.pos_y = 0 ; 
	mouse_state.left_button_clicked = false ;
	mouse_state.left_button_released = true ; 
	mouse_state.right_button_clicked = false ;
	mouse_state.right_button_released = true ; 
	mouse_state.previous_pos_x = 0 ; 
	mouse_state.previous_pos_y = 0 ;  	
	default_framebuffer_id = 0 ; 
	light_database = new LightingDatabase(); 
	texture_database = TextureDatabase::getInstance();
	shader_database = ShaderDatabase::getInstance();	
	Loader::loadShaderDatabase();
	scene_camera = new ArcballCamera(45.f , &screen_size ,  0.1f , 10000.f , 100.f, &mouse_state);
	camera_framebuffer = new CameraFrameBuffer(texture_database , shader_database , &screen_size , &default_framebuffer_id);  
	scene = new Scene(); 
}

Renderer::Renderer(unsigned width , unsigned height):Renderer(){
	screen_size.width = width ; 
	screen_size.height = height ;
	
}

Renderer::~Renderer(){
	if(scene != nullptr){
		scene->clear(); 
		delete scene ; 
	}
	if(TextureDatabase::isInstanced()){
		texture_database->hardCleanse();
		texture_database->destroy();   
		texture_database = nullptr ; 
	}
	if(ShaderDatabase::isInstanced()){
		shader_database->clean();
		shader_database->destroy();
		shader_database = nullptr ; 
	}
	if(light_database){
		light_database->clearDatabase(); 
		delete light_database;
	}
	if(camera_framebuffer){
		camera_framebuffer->clean();
		delete camera_framebuffer; 
		camera_framebuffer = nullptr;  
	}
	delete scene_camera ; 
	scene_camera = nullptr ; 
}

void Renderer::initialize(){
	glEnable(GL_DEPTH_TEST);	
	camera_framebuffer->initializeFrameBuffer(); 	
}

bool Renderer::scene_ready(){
	if(!scene->isReady())
		return false; 
	if(camera_framebuffer && !camera_framebuffer->getDrawable()->ready())
		return false; 
	return true ; 
}

bool Renderer::prep_draw(){	
	if(start_draw && scene_ready()){	
		camera_framebuffer->startDraw();
		scene->prepare_draw(scene_camera); 			
		return true; 				
	}
	else{
		glClearColor(0 , 0 , 0.1 , 1.f);
		return false ;	
	}
}

void Renderer::draw(){	
	scene_camera->computeViewProjection();		
	camera_framebuffer->bindFrameBuffer();	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	std::vector<Drawable*> transparent_meshes = scene->getSortedTransparentElements();  
	std::vector<Drawable*> opaque_meshes = scene->getOpaqueElements();
	for(Drawable *A : opaque_meshes){
		A->bind(); 
		light_database->updateShadersData(A->getMeshShaderPointer() , A->getMeshPointer()->getModelViewMatrix()); 
		glDrawElements(GL_TRIANGLES , A->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		A->unbind();
	}
	for(Drawable *A : transparent_meshes){
		A->bind();	
		light_database->updateShadersData(A->getMeshShaderPointer() , A->getMeshPointer()->getModelViewMatrix()); 
		glDrawElements(GL_TRIANGLES , A->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
		A->unbind();
	}

	camera_framebuffer->unbindFrameBuffer();	
	camera_framebuffer->renderFrameBufferMesh();	
}

void Renderer::set_new_scene(std::vector<Mesh*> &new_scene){
	scene->clear(); 
	scene->setScene(new_scene);  	
	start_draw = true ;
	scene_camera->reset() ;
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
void Renderer::onScrollDown(){
	scene_camera->zoomOut(); 
}
void Renderer::onScrollUp(){
	scene_camera->zoomIn(); 
}

void Renderer::onResize(unsigned int width , unsigned int height){
	screen_size.width = width; 
	screen_size.height = height;
	if(camera_framebuffer)
		camera_framebuffer->resize();  
}





