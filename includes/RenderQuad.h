#ifndef RENDERQUAD_H
#define RENDERQUAD_H

#include "FrameBufferInterface.h"

class RenderQuadFBO : public FrameBufferInterface{
public:
    RenderQuadFBO();
    RenderQuadFBO(TextureDatabase *texture_database , ScreenSize* texture_size , unsigned int *default_fbo_id); 
    virtual ~RenderQuadFBO(); 
    virtual void renderToTexture(GLFrameBuffer::INTERNAL_FORMAT internal_format = GLFrameBuffer::COLOR0); 


};





#endif