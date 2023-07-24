#include "../includes/Renderer.h"
#include "../includes/Loader.h"
#include "../includes/RenderPipeline.h"

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

Renderer::Renderer(unsigned width , unsigned height , GLViewer* widget):Renderer(){
	screen_size.width = width ; 
	screen_size.height = height ;
	gl_widget = widget ; 
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
	if(render_pipeline){
		render_pipeline->clean();
		delete render_pipeline;
		render_pipeline = nullptr ;
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
	render_pipeline = new RenderPipeline(this , resource_database); 
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
	scene->updateTree(); 
	camera_framebuffer->bindFrameBuffer();	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		
	scene->drawForwardTransparencyMode(); 
	scene->drawBoundingBoxes();	
	camera_framebuffer->unbindFrameBuffer();	
	camera_framebuffer->renderFrameBufferMesh();
	errorCheck(__FILE__, __LINE__);  
}

void Renderer::set_new_scene(std::pair<std::vector<Mesh*> , SceneTree> &new_scene){
	scene->clear();	
	Loader loader; 
	EnvironmentMapTexture* env = loader.loadHdrEnvmap();  //! TODO in case we want to seek the cubemap to replace it's texture with this , use visitor pattern in scene graph 
	//CubeMapMesh* cubemap_mesh = render_pipeline->bakeEnvmapToCubemap(dynamic_cast<EnvironmentMapTexture*>(resource_database->getTextureDatabase()->getTexturesByType(Texture::ENVMAP)[0].second), 1024 , 1024 , gl_widget); 
	CubeMapMesh* cubemap_mesh = render_pipeline->bakeEnvmapToCubemap(env , 2048 , 2048  , gl_widget);
	int irradiance_tex_id = resource_database->getTextureDatabase()->getTexturesByType(Texture::IRRADIANCE)[0].first; 
	std::for_each(new_scene.first.begin() , new_scene.first.end() , [&irradiance_tex_id](Mesh* m){m->material.addTexture(irradiance_tex_id , Texture::IRRADIANCE) ; });
	assert(cubemap_mesh);
	new_scene.first.push_back(cubemap_mesh); 
	new_scene.second.setAsRootChild(cubemap_mesh); 
	scene->setScene(new_scene); 	
	scene->setLightDatabasePointer(light_database); 
	scene->setCameraPointer(scene_camera); 
	light_database->clearDatabase();
	AbstractLight *L1 = new SpotLight(glm::vec3(-55 , 90 , -5) , glm::vec3(0.f) , glm::vec3(1.f , 1.f , 0.9f), 12.f , 102000.f , scene_camera); 
    AbstractLight *L2 = new PointLight(glm::vec3(0 , 20 , -2) , glm::vec3(0.875f , 0.257f , 0.184f), glm::vec3(1.f , 0.0045 , 0.0075) , 200.f , L1); 
	AbstractLight *L3 = new DirectionalLight(glm::vec3(0 , 1 , 0) , glm::vec3(1.f , 1.f , 1.f) , .5f , scene_camera); 
	light_database->addLight(L1); 
	light_database->addLight(L2);
	light_database->addLight(L3); 
	scene->generateBoundingBoxes(resource_database->getShaderDatabase()->get(Shader::BOUNDING_BOX)); 		
	start_draw = true ;
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

void Renderer::setGammaValue(float value){
	if(camera_framebuffer != nullptr){
		camera_framebuffer->setGamma(value);
		gl_widget->update(); 
	} 
}

void Renderer::setExposureValue(float value){
	if(camera_framebuffer != nullptr){
		camera_framebuffer->setExposure(value); 
		gl_widget->update();
	}
}

void Renderer::setNoPostProcess(){
	if(camera_framebuffer != nullptr){
		camera_framebuffer->setPostProcessDefault(); 
		gl_widget->update(); 
	}
}

void Renderer::setPostProcessEdge(){
	if(camera_framebuffer != nullptr){
		camera_framebuffer->setPostProcessEdge(); 
		gl_widget->update(); 
	}
} 

void Renderer::setPostProcessSharpen(){
	if(camera_framebuffer != nullptr){
		camera_framebuffer->setPostProcessSharpen(); 
		gl_widget->update(); 
	}
} 

void Renderer::setPostProcessBlurr(){
	if(camera_framebuffer != nullptr){
		camera_framebuffer->setPostProcessBlurr(); 
		gl_widget->update(); 
	}
} 

void Renderer::resetSceneCamera(){
	if(scene_camera != nullptr){
		scene_camera->reset(); 
		gl_widget->update(); 
	}
} 

void Renderer::setRasterizerFill(){
	if(scene){
		scene->setPolygonFill(); 
		gl_widget->update();
	}
}

void Renderer::setRasterizerPoint(){
	if(scene){
		scene->setPolygonPoint(); 
		gl_widget->update();
	}
}

void Renderer::setRasterizerWireframe(){
	if(scene){
		scene->setPolygonWireframe(); 
		gl_widget->update();
	}
}

void Renderer::displayBoundingBoxes(bool display){
	if(scene){
		scene->displayBoundingBoxes(display); 
		gl_widget->update();
	}
}