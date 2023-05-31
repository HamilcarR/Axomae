#ifndef GLBufferInterface_H
#define GLBufferInterface_H

#include "utils_3D.h" 

/**
 * @file GLBufferInterface.h
 * Defines an interface for Opengl buffer wrappers
 * 
 */

/**
 * @brief Interface for opengl function wrappers
 * @class GLBufferInterface
 */
class GLBufferInterface{
public:
    
    /**
     * @brief Initializes the buffers the class uses
     * 
     */
    virtual void initializeBuffers() = 0 ;
    
    /**
     * @brief Checks if buffers IDs have been initialized
     * 
     * @return true If ID is ready to use
     */
    virtual bool isReady() const = 0 ;

    /**
     * @brief Fills the buffers with raw data
     * 
     */
    virtual void fillBuffers() = 0 ;

    /**
     * @brief Generic method to bind the GLBuffer class object
     * 
     */
    virtual void bind() = 0 ; 

    /**
     * @brief Generic method to unbind the GLBuffer class object 
     * 
     */
    virtual void unbind() = 0 ;  

    /**
     * @brief Free the IDs used by OpenGL
     * 
     */
    virtual void clean() = 0 ;  


};











#endif