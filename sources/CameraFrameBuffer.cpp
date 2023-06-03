#include "../includes/CameraFrameBuffer.h"
#include "../includes/UniformNames.h"


using namespace axomae ; 


int setUpEmptyTexture(TextureDatabase* database, unsigned width , unsigned height){    
    TextureData temp_empty_data_texture ; 
    temp_empty_data_texture.width = width ; 
    temp_empty_data_texture.height = height ;
    temp_empty_data_texture.data = nullptr;   
    return database->addTexture(&temp_empty_data_texture , Texture::FRAMEBUFFER , true); //TODO fix bug on last argument : keep_texture_after_clean 
}

CameraFrameBuffer::CameraFrameBuffer(TextureDatabase* _texture_database , ShaderDatabase* _shader_database, ScreenSize* screen_size_pointer , unsigned int* default_fbo_pointer){
    screen_dim = screen_size_pointer ; 
    default_framebuffer_pointer = default_fbo_pointer;  
    texture_database = _texture_database ;
    shader_database = _shader_database ;
    gamma = 1.2f; 
    screen_texture_database_id = 0 ; 
    gl_framebuffer_object = nullptr ; 
    drawable_screen_quad = nullptr; 
    mesh_screen_quad = nullptr; 
    shader_framebuffer = nullptr ;
    fbo_texture_pointer = nullptr;  
}

CameraFrameBuffer::~CameraFrameBuffer(){
    if(drawable_screen_quad)
        delete drawable_screen_quad ;
    if(gl_framebuffer_object != nullptr)
        delete gl_framebuffer_object ;  
}

void CameraFrameBuffer::initializeFrameBuffer(){  
    screen_texture_database_id = setUpEmptyTexture(texture_database , screen_dim->width , screen_dim->height);
    shader_framebuffer = shader_database->get(Shader::SCREEN_FRAMEBUFFER);
    fbo_texture_pointer = texture_database->get(screen_texture_database_id); 
    assert(fbo_texture_pointer != nullptr);  
    mesh_screen_quad = new FrameBufferMesh(screen_texture_database_id , shader_framebuffer) ;
    drawable_screen_quad = new Drawable(mesh_screen_quad) ; 
    gl_framebuffer_object = new GLFrameBuffer(screen_dim->width , screen_dim->height, fbo_texture_pointer->getSamplerID() , GLFrameBuffer::COLOR0 , GLRenderBuffer::DEPTH24_STENCIL8 , default_framebuffer_pointer); 
    gl_framebuffer_object->initializeBuffers();
}

void CameraFrameBuffer::clean(){
    if(drawable_screen_quad)
        drawable_screen_quad->clean();
    if(gl_framebuffer_object)
        gl_framebuffer_object->clean(); 

}   

void CameraFrameBuffer::bindFrameBuffer(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->bind(); 
}

void CameraFrameBuffer::unbindFrameBuffer(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->unbind(); 
}

void CameraFrameBuffer::startDraw(){
    if(shader_framebuffer){
        shader_framebuffer->setUniform(uniform_name_float_gamma_name , gamma); 
    }
    if(drawable_screen_quad)
        drawable_screen_quad->startDraw(); 
}

void CameraFrameBuffer::renderFrameBufferMesh(){
    drawable_screen_quad->bind();
	glDrawElements(GL_TRIANGLES , drawable_screen_quad->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
	drawable_screen_quad->unbind();

}

void CameraFrameBuffer::resize(){
    if(fbo_texture_pointer && screen_dim && gl_framebuffer_object){
        fbo_texture_pointer->setNewSize(screen_dim->width , screen_dim->height); 
        gl_framebuffer_object->resize(screen_dim->width , screen_dim->height); 
    }
}