#ifndef RENDERCUBEMAP_H
#define RENDERCUBEMAP_H

#include "FrameBufferInterface.h"
#include "Shader.h"


/**
 * @brief A Framebuffer that renders to a cubemap
 *  
 */
class RenderCubeMap : public FrameBufferInterface{
public:
    RenderCubeMap();
    RenderCubeMap(TextureDatabase* texture_database , ScreenSize* texture_size , unsigned int* default_fbo_pointer_id); 
    virtual ~RenderCubeMap(); 
    virtual void renderToFace(unsigned i = 0 , GLFrameBuffer::INTERNAL_FORMAT color_attachment = GLFrameBuffer::COLOR0 );
    virtual int setUpEmptyTexture(unsigned width , unsigned height , Texture::TYPE type); 


protected:



};



#endif