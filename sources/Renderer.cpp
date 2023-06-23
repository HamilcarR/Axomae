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
	resource_database = ResourceDatabaseManager::getInstance();
	Loader loader ; 
	loader.loadShaderDatabase();
	scene_camera = new ArcballCamera(45.f , &screen_size ,  0.1f , 10000.f , 100.f, &mouse_state);
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
	if(resource_database){
		resource_database->purge();
		resource_database->destroyInstance(); 
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
	resource_database->getShaderDatabase()->initializeShaders(); 		
	camera_framebuffer = new CameraFrameBuffer(resource_database->getTextureDatabase() , resource_database->getShaderDatabase() , &screen_size , &default_framebuffer_id);  
	camera_framebuffer->initializeFrameBuffer(); 	
	scene = new Scene(); 
	
}

bool Renderer::scene_ready(){
	if(!scene->isReady())
		return false; 
	if(camera_framebuffer && !camera_framebuffer->getDrawable()->ready())
		return false; 
	return true ; 
}

/*This function is executed each frame*/
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
	camera_framebuffer->bindFrameBuffer();	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		
	scene->drawForwardTransparencyMode(); 
	scene->drawBoundingBoxes();	
	camera_framebuffer->unbindFrameBuffer();	
	camera_framebuffer->renderFrameBufferMesh();	
}

void Renderer::set_new_scene(std::vector<Mesh*> &new_scene){
	scene->clear();
	scene->setScene(new_scene);	
	scene->setLightDatabasePointer(light_database);   
	scene->generateBoundingBoxes(resource_database->getShaderDatabase()->get(Shader::BOUNDING_BOX)); 	
	scene->setCameraPointer(scene_camera); 
	start_draw = true ;
	scene_camera->reset() ;
	resource_database->getShaderDatabase()->initializeShaders(); 
	camera_framebuffer->updateFrameBufferShader(); 
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





