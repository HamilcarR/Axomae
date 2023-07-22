#include "../includes/GLRenderBuffer.h"


GLRenderBuffer::GLRenderBuffer(){
    width = 0 ; 
    height = 0 ; 
    format = EMPTY; 
}

GLRenderBuffer::~GLRenderBuffer(){

}

GLRenderBuffer::GLRenderBuffer(unsigned int _width , unsigned int _height , INTERNAL_FORMAT _format){
    width = _width; 
    height = _height ; 
    format = _format ; 
}

void GLRenderBuffer::initializeBuffers(){
    glGenRenderbuffers(1 , &renderbuffer_id);
    bind();  
    glRenderbufferStorage(GL_RENDERBUFFER , format , width , height);
    errorCheck(__FILE__ , __LINE__); 
}

bool GLRenderBuffer::isReady() const {
    return renderbuffer_id != 0; 
}

void GLRenderBuffer::fillBuffers(){

}

void GLRenderBuffer::bind(){
    glBindRenderbuffer(GL_RENDERBUFFER , renderbuffer_id); 
}

void GLRenderBuffer::unbind(){
    glBindRenderbuffer(GL_RENDERBUFFER , 0); 
}

void GLRenderBuffer::clean(){
    unbind(); 
    glDeleteRenderbuffers(1 , &renderbuffer_id); 
}

void GLRenderBuffer::resize(unsigned _width , unsigned _height){
    bind(); 
    glRenderbufferStorage(GL_RENDERBUFFER , format , _width , _height);
    unbind();  
    width = _width ; 
    height = _height ; 
}