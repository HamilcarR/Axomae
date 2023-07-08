#ifndef CAMERAFRAMEBUFFER_H
#define CAMERAFRAMEBUFFER_H

#include "FrameBufferInterface.h"


/**
 * @file CameraFrameBuffer.h
 * This file implements the post processing framebuffer system
 */


/**
 * @class CameraFrameBuffer
 * This class implements post processing effects 
 */
class CameraFrameBuffer : public FrameBufferInterface{
public:

    /**
     * @brief Construct a new Camera Frame Buffer object
     * 
     * @param texture_database Pointer to the database of textures
     * @param shader_database Pointer to the shader database
     * @param screen_size_pointer Pointer on a ScreenSize structure , containing informations about the dimensions of the render surface
     * @param default_fbo_id Pointer on the ID of the default framebuffer . In the case of QT , this framebuffer is the one used for the GUI interface (so , not 0) 
     */
    CameraFrameBuffer(TextureDatabase* texture_database ,ShaderDatabase* shader_database,  ScreenSize* screen_size_pointer , unsigned int *default_fbo_id);  
    
    /**
     * @brief Destroy the Camera Frame Buffer object
     * 
     */
    virtual ~CameraFrameBuffer(); 
    
    /**
     * @brief Renders the quad mesh of the framebuffer
     * 
     */
    virtual void renderFrameBufferMesh(); 

    /**
     * @brief Returns a pointer on the drawable of the quad mesh
     * 
     * @return Drawable* 
     */
    Drawable* getDrawable(){return drawable_screen_quad;}
    
    /**
     * @brief Initializes the textures used by the framebuffer , the shader , and creates the quad mesh that the framebuffer will draw on
     * 
     */
    virtual void initializeFrameBuffer() override; 

    /**
     * @brief Send the uniforms used by the post processing effects , like gamma and exposure , and sets up the mesh used
     * 
     */
    virtual void startDraw();

    /**
     * @brief Clean the post processing structure entirely , freeing the ressources 
     * 
     */
    virtual void clean() override;

    /**
     * @brief Update the shader used for post processing 
     * 
     */
    virtual void updateFrameBufferShader();

private:

    /**
     * @brief Construct a new Camera Frame Buffer object
     * 
     */
    CameraFrameBuffer(); 


protected: 
    Drawable *drawable_screen_quad;         /*<Drawable of the screen quad*/ 
    Mesh *mesh_screen_quad;                 /*<Pointer on the screen quad mesh*/
    ShaderDatabase* shader_database;        /*<Pointer on the shader database*/
    Shader* shader_framebuffer;             /*<Post processing shader*/
    float gamma ;                           /*<Gamma of the screen*/
    float exposure ;                        /*<Exposure of the screen*/
    
}; 















#endif