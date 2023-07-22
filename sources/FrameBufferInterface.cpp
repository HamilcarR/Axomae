#include "../includes/FrameBufferInterface.h"

//!Reimplement in child classes , put "keep" to false
int FrameBufferInterface::setUpEmptyTexture( unsigned width , unsigned height , Texture::TYPE type){    
    TextureData temp_empty_data_texture ; 
    temp_empty_data_texture.width = width ; 
    temp_empty_data_texture.height = height ;
    temp_empty_data_texture.data = nullptr; 
    temp_empty_data_texture.internal_format = internal_format ; 
    temp_empty_data_texture.data_format = data_format ; 
    temp_empty_data_texture.data_type = data_type ;   
    return texture_database->addTexture(&temp_empty_data_texture , type , true); 
}

FrameBufferInterface::FrameBufferInterface(Texture::TYPE type){
    gl_framebuffer_object = nullptr; 
    render_type = type ; 
    texture_dim = nullptr; 
    texture_database = nullptr; 
    fbo_texture_pointer = nullptr; 
    default_framebuffer_pointer = nullptr;   
    texture_id = 0 ; 
    fbo_texture_pointer = nullptr; 
}

FrameBufferInterface::FrameBufferInterface(TextureDatabase* _texture_database ,
                                        ScreenSize* _texture_dim ,
                                        unsigned int* default_fbo ,
                                        GLFrameBuffer::INTERNAL_FORMAT _color_attachment , 
                                        Texture::TYPE _render_type , 
                                        Texture::FORMAT _internal_format  ,
                                        Texture::FORMAT _data_format  ,
                                        Texture::FORMAT _data_type   ) : FrameBufferInterface(_render_type){
    texture_dim = _texture_dim ; 
    texture_database = _texture_database;
    default_framebuffer_pointer = default_fbo ; 
    internal_format = _internal_format; 
    color_attachment = _color_attachment ; 
    data_format = _data_format ; 
    data_type = _data_type ; 
    assert(texture_dim != nullptr); 
}


FrameBufferInterface::~FrameBufferInterface(){
    if(gl_framebuffer_object != nullptr)
        delete gl_framebuffer_object ;  
}

void FrameBufferInterface::resize(){
    if(fbo_texture_pointer && texture_dim && gl_framebuffer_object){
        fbo_texture_pointer->setNewSize(texture_dim->width , texture_dim->height); 
        gl_framebuffer_object->resize(texture_dim->width , texture_dim->height); 
    }
}

void FrameBufferInterface::bindFrameBuffer(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->bind(); 
}

void FrameBufferInterface::unbindFrameBuffer(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->unbind(); 
}

void FrameBufferInterface::clean(){
    if(gl_framebuffer_object)
        gl_framebuffer_object->clean(); 
}

void FrameBufferInterface::initializeFrameBufferTexture(){
    texture_id = setUpEmptyTexture(texture_dim->width , texture_dim->height , render_type);
    fbo_texture_pointer = texture_database->get(texture_id); 

}

void FrameBufferInterface::initializeFrameBuffer(){   
    gl_framebuffer_object = new GLFrameBuffer(texture_dim->width , texture_dim->height, fbo_texture_pointer->getSamplerID() ,  GLRenderBuffer::DEPTH24_STENCIL8 , default_framebuffer_pointer); 
    gl_framebuffer_object->initializeBuffers();

}