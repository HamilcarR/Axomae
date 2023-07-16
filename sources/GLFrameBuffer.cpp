#include "../includes/GLFrameBuffer.h"


GLFrameBuffer::GLFrameBuffer(){
    renderbuffer_object = nullptr;
    framebuffer_id = 0 ; 
    texture_id = 0 ; 
    format = COLOR0;
}

GLFrameBuffer::GLFrameBuffer(unsigned _width , unsigned _height , unsigned _texture_id , INTERNAL_FORMAT fbo_format , GLRenderBuffer::INTERNAL_FORMAT rbo_format , unsigned int* default_fbo_id_pointer):GLFrameBuffer(){
    if(rbo_format != GLRenderBuffer::EMPTY)
        renderbuffer_object = new GLRenderBuffer(_width, _height , rbo_format); 
    format = fbo_format ;
    texture_id = _texture_id ;
    pointer_on_default_fbo_id = default_fbo_id_pointer ;  
}

GLFrameBuffer::~GLFrameBuffer(){
    if(renderbuffer_object != nullptr)
        delete renderbuffer_object; 
}

void GLFrameBuffer::initializeBuffers(){
    glGenFramebuffers(1 , &framebuffer_id);
    bind(); 
    glFramebufferTexture2D(GL_FRAMEBUFFER , format , GL_TEXTURE_2D , texture_id , 0);
    if(renderbuffer_object != nullptr){
        renderbuffer_object->initializeBuffers(); 
        if(renderbuffer_object->isReady()){
            renderbuffer_object->bind(); 
            glFramebufferRenderbuffer(GL_FRAMEBUFFER , DEPTH_STENCIL , GL_RENDERBUFFER , renderbuffer_object->getID());
        }
        else
            std::cout << "Problem initializing render buffer" << "\n";     
        auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE){
            std::cout << "Framebuffer not ready to use " << std::to_string(status) << "\n";  
        }
    }
    unbind();        
}

bool GLFrameBuffer::isReady() const {
    return framebuffer_id != 0 ; 
}

void GLFrameBuffer::fillBuffers(){

}

void GLFrameBuffer::bind(){
    glBindFramebuffer(GL_FRAMEBUFFER , framebuffer_id); 
}

void GLFrameBuffer::unbind(){
    if(pointer_on_default_fbo_id)
        glBindFramebuffer(GL_FRAMEBUFFER , *pointer_on_default_fbo_id);  
    else
        glBindFramebuffer(GL_FRAMEBUFFER , 0); 
}

void GLFrameBuffer::clean(){
    if(renderbuffer_object != nullptr){
        glFramebufferRenderbuffer(GL_FRAMEBUFFER , DEPTH_STENCIL , GL_RENDERBUFFER , 0);
        renderbuffer_object->clean(); 
    }
    unbind(); 
    glDeleteFramebuffers(1 , &framebuffer_id); 
}

void GLFrameBuffer::resize(unsigned _width , unsigned _height){
    if(renderbuffer_object)
        renderbuffer_object->resize(_width , _height); 
} 