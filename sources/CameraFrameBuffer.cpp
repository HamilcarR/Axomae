#include "../includes/CameraFrameBuffer.h"
#include "../includes/UniformNames.h"


using namespace axomae ; 




CameraFrameBuffer::CameraFrameBuffer(TextureDatabase* _texture_database , 
                                    ShaderDatabase* _shader_database, 
                                    ScreenSize* screen_size_pointer , 
                                    unsigned int* default_fbo_pointer) : FrameBufferInterface(_texture_database , screen_size_pointer , default_fbo_pointer){
    internal_format = Texture::RGBA16F ; 
    data_format = Texture::BGRA ; 
    data_type = Texture::UBYTE ; 
    texture_database = _texture_database ;
    shader_database = _shader_database ;
    gamma = 1.2f;
    exposure = 0.3f;  
    texture_id = 0 ; 
    gl_framebuffer_object = nullptr ; 
    drawable_screen_quad = nullptr; 
    mesh_screen_quad = nullptr; 
    shader_framebuffer = nullptr ;
    fbo_texture_pointer = nullptr;  
}

CameraFrameBuffer::~CameraFrameBuffer(){
    if(drawable_screen_quad)
        delete drawable_screen_quad ;
    
}


void CameraFrameBuffer::updateFrameBufferShader(){
    shader_framebuffer = static_cast<ScreenFrameBufferShader*>(shader_database->get(Shader::SCREEN_FRAMEBUFFER)); 
    assert(mesh_screen_quad); 
    mesh_screen_quad->setShader(shader_framebuffer); 
}

void CameraFrameBuffer::initializeFrameBuffer(){  
    initializeFrameBufferTexture(); 
    shader_framebuffer = static_cast<ScreenFrameBufferShader*>(shader_database->get(Shader::SCREEN_FRAMEBUFFER));
    mesh_screen_quad = new FrameBufferMesh(texture_id , shader_framebuffer) ;
    drawable_screen_quad = new Drawable(mesh_screen_quad) ; 
    FrameBufferInterface::initializeFrameBuffer(); 
    bindFrameBuffer(); 
    gl_framebuffer_object->attachTexture2D(GLFrameBuffer::TEXTURE2D );
    unbindFrameBuffer();  
}

void CameraFrameBuffer::clean(){
    if(drawable_screen_quad)
        drawable_screen_quad->clean();
    FrameBufferInterface::clean(); 
}   


void CameraFrameBuffer::startDraw(){
    if(shader_framebuffer){
        shader_framebuffer->bind(); 
        shader_framebuffer->setUniform(uniform_name_float_gamma_name , gamma);
        shader_framebuffer->setUniform(uniform_name_float_exposure_name , exposure); 
        shader_framebuffer->setPostProcessUniforms();  
        shader_framebuffer->release(); 
    }
    if(drawable_screen_quad)
        drawable_screen_quad->startDraw(); 
}

void CameraFrameBuffer::renderFrameBufferMesh(){
    drawable_screen_quad->bind();
	glDrawElements(GL_TRIANGLES , drawable_screen_quad->getMeshPointer()->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
	drawable_screen_quad->unbind();
}

void CameraFrameBuffer::setPostProcessEdge(){
    if(shader_framebuffer)
        shader_framebuffer->setPostProcess(ScreenFrameBufferShader::EDGE); 
}

void CameraFrameBuffer::setPostProcessDefault(){
    if(shader_framebuffer)
        shader_framebuffer->setPostProcess(ScreenFrameBufferShader::DEFAULT); 
}

void CameraFrameBuffer::setPostProcessBlurr(){
    if(shader_framebuffer)
        shader_framebuffer->setPostProcess(ScreenFrameBufferShader::BLURR); 
}

void CameraFrameBuffer::setPostProcessSharpen(){
    if(shader_framebuffer)
        shader_framebuffer->setPostProcess(ScreenFrameBufferShader::SHARPEN); 
}