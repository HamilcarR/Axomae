#include "../includes/Drawable.h"


using namespace axomae ;


Drawable::Drawable(){
	mesh_object = nullptr; 
	camera_pointer = nullptr ; 
}

Drawable::Drawable(Mesh &mesh){
	mesh_object = new Mesh(mesh);
	camera_pointer = nullptr ;
	gl_buffers.setGeometryPointer(&mesh_object->geometry) ; 
	if(!initialize()){
		std::cerr << "failed initializing Drawable data" << "\n" ; 
		exit(EXIT_FAILURE); 
	}
}

Drawable::~Drawable(){

}

void Drawable::clean(){
	gl_buffers.clean() ; 
	mesh_object->clean(); 	

}


bool Drawable::ready(){
	return gl_buffers.isReady() ; 
}

bool Drawable::initialize(){
	if(mesh_object == nullptr)
		return false ;
	mesh_object->initializeGlData(); 
	gl_buffers.initializeBuffers(); 
	errorCheck(); 
	return gl_buffers.isReady();  ; 
}

void Drawable::start_draw(){
	if(mesh_object != nullptr){		
		mesh_object->bindShaders();
		gl_buffers.bindVao() ;  
		gl_buffers.fillBuffers() ; 	
		
		gl_buffers.bindVertexBuffer(); 
		mesh_object->shader_program.enableAttributeArray(0);
		mesh_object->shader_program.setAttributeBuffer(0 , GL_FLOAT , 0 , 3 , 0 ) ;
		
		gl_buffers.bindColorBuffer() ;
		mesh_object->shader_program.enableAttributeArray(1);
		mesh_object->shader_program.setAttributeBuffer(1 , GL_FLOAT , 0 , 3 , 0 ) ; 
	
		gl_buffers.bindNormalBuffer() ;
		mesh_object->shader_program.enableAttributeArray(2);
		mesh_object->shader_program.setAttributeBuffer(2 , GL_FLOAT , 0 , 3 , 0 ) ; 
		
		gl_buffers.bindTextureBuffer() ;
		mesh_object->shader_program.enableAttributeArray(3);
		mesh_object->shader_program.setAttributeBuffer(3 , GL_FLOAT , 0 , 2 , 0 ) ; 

		gl_buffers.bindTangentBuffer() ;
		mesh_object->shader_program.enableAttributeArray(4);
		mesh_object->shader_program.setAttributeBuffer(4 , GL_FLOAT , 0 , 3 , 0 ) ; 
	
		gl_buffers.unbindVao() ; 
	}

}



void Drawable::bind(){
	mesh_object->bindShaders(); 
	mesh_object->bindMaterials(); 
	gl_buffers.bindVao(); 

}

void Drawable::unbind(){
	gl_buffers.unbindVao(); 
	mesh_object->releaseShaders();
}

void Drawable::setSceneCameraPointer(Camera* camera){
	camera_pointer = camera ;
	mesh_object->setSceneCameraPointer(camera); 
}





