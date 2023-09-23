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
	camera_framebuffer->clean();
	scene->clear(); 
	render_pipeline->clean();
	
	if(resource_database){
		resource_database->purge();
		resource_database->destroyInstance(); 
	}
	if(light_database){
		light_database->clearDatabase(); 
		delete light_database;
	}
		
	
	delete scene_camera ; 
	scene_camera = nullptr ; 
}

void Renderer::initialize(){
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS); 		
	resource_database->getShaderDatabase()->initializeShaders(); 		
	camera_framebuffer = std::make_unique<CameraFrameBuffer>(resource_database->getTextureDatabase() , resource_database->getShaderDatabase() , &screen_size , &default_framebuffer_id);  
	render_pipeline = std::make_unique<RenderPipeline>(this , resource_database); 
	scene = std::make_unique<Scene>(); 
	camera_framebuffer->initializeFrameBuffer(); 	
	
	
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
	EnvironmentMap2DTexture* env = loader.loadHdrEnvmap();  //! TODO in case we want to seek the cubemap to replace it's texture with this , use visitor pattern in scene graph 
	CubeMapMesh* cubemap_mesh = render_pipeline->bakeEnvmapToCubemap(env , 2048 , 2048  , gl_widget);
	int cube_envmap_id =  cubemap_mesh->material.getTextureGroup().getTextureCollection()[0] ;
	int irradiance_tex_id = render_pipeline->bakeIrradianceCubemap(cube_envmap_id , 64 , 64 , gl_widget);
	int prefiltered_cubemap = render_pipeline->preFilterEnvmap(cube_envmap_id , 2048 , 512 , 512 , 10 , 2000 , 2 , gl_widget);
	int brdf_lut = render_pipeline->generateBRDFLookupTexture(512 , 512 , gl_widget);  
	std::for_each(new_scene.first.begin() , new_scene.first.end() , [irradiance_tex_id , brdf_lut , prefiltered_cubemap , cube_envmap_id , cubemap_mesh](Mesh* m){
																						m->material.addTexture(irradiance_tex_id , Texture::IRRADIANCE) ;
																						m->material.addTexture(prefiltered_cubemap , Texture::CUBEMAP); 
																						//m->material.addTexture(cube_envmap_id , Texture::CUBEMAP); 
																						m->material.addTexture(brdf_lut , Texture::BRDFLUT);
																						m->setCubemapPointer(cubemap_mesh);
																						});
	assert(cubemap_mesh);
	new_scene.first.push_back(cubemap_mesh); 
	new_scene.second.pushNewRoot(cubemap_mesh); 
	scene->setScene(new_scene); 	
	scene->setLightDatabasePointer(light_database); 
	scene->setCameraPointer(scene_camera); 
	light_database->clearDatabase();
	AbstractLight *L1 = new SpotLight(glm::vec3(-55 , 90 , -5) , glm::vec3(0.f) , glm::vec3(1.f , 1.f , 0.9f), 12.f , 102000.f , scene_camera); 
    AbstractLight *L2 = new PointLight(glm::vec3(0 , 20 , -2) , glm::vec3(0.875f , 0.257f , 0.184f), glm::vec3(1.f , 0.0045 , 0.0075) , 200.f , L1); 
	AbstractLight *L3 = new DirectionalLight(glm::vec3(1 , 1 , 0) , glm::vec3(1.f , 1.f , 1.f) , 1.5f , scene_camera); 
	//light_database->addLight(L1); 
	//light_database->addLight(L2);
	//light_database->addLight(L3); 
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

