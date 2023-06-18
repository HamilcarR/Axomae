#ifndef SCENE_H
#define SCENE_H

#include "utils_3D.h"
#include "Drawable.h"
#include "BoundingBox.h"
#include "Camera.h"

/**
 * @file Scene.h
 * File implementing classes and functions relative to how the scene is represented and how to manage it 
 * 
 */



class Scene{
public:

    struct AABB{
        BoundingBox aabb; 
        Drawable* drawable; 
    }; 

    /**
     * @brief Construct a new Scene object
     * 
     */
    Scene(); 
    
    /**
     * @brief Destroy the Scene object
     * 
     */
    virtual ~Scene(); 
    
    /**
     * @brief Sets the scene
     * 
     * @param to_copy Meshes to copy  
     */
    void setScene(std::vector<axomae::Mesh*>& to_copy); 
    
    /**
     * @brief Get only the opaque objects. These objects are not sorted , since the draw order doesn't have an incidence
     * 
     * @return std::vector<Drawable*> Array of all opaque elements 
     */
    virtual std::vector<Drawable*> getOpaqueElements() const ; 
     
    /**
    * @brief This method returns a vector of transparent meshes sorted in reverse order based on their
    * distance from the camera.
    * 
    * @return A vector of pointers to Drawable objects, representing the sorted transparent elements in
    * the Scene.
    */
    virtual std::vector<Drawable*> getSortedTransparentElements();

    /**
     * @brief This method returns an array of bounding boxes of each mesh in the scene. 
     * Note that special meshes like the screen framebuffer , the cubemap , light sprites are not concerned by this method.
     * 
     * @return std::vector<Drawable*> Array of all meshes bounding boxes in the scene
     */
    virtual std::vector<Drawable*> getBoundingBoxElements();

    /**
     * @brief This method will add bounding boxes meshes to the scene. 
     * @param box_shader Shader responsible for displaying bounding boxes
     */
    virtual void generateBoundingBoxes(Shader* box_shader); 

    /**
    * The function clears the scene by deleting all drawables and clearing the scene and
    * sorted_transparent_meshes vectors.
    */
    void clear();
 
    /**
    * @brief The function checks if all objects in the scene are ready to be drawn.
    * 
    * @return The function `isReady()` is returning a boolean value. It returns `true` if all the objects
    * in the `scene` are drawable and ready, and `false` otherwise.
    */
    bool isReady();

    /**
     * @brief Set up the scene camera , and prep drawables for the next draw. 
     * 
     * @param camera Pointer on the scene camera
     */
    void prepare_draw(Camera* camera);  


private:

    /**
     * @brief Sort transparent elements by distance and store their position in sorted_transparent_meshes
     * 
     */
    void sortTransparentElements(); 
protected:
    std::map<float , Drawable*> sorted_transparent_meshes ; 
    std::vector<AABB> scene ;
    std::vector<Drawable*> bounding_boxes_array;

};























#endif