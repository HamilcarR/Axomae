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


//TODO: [AX-14] Add mouse picking 
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

    /**
     * @brief This method sorts the scene according to the meshes transparency ... opaque objects are first , and transparent objects are sorted according to distance. 
     * 
     * @return std::vector<Drawable*> 
     */
    std::vector<Drawable*> getSortedSceneByTransparency();

    /**
     * @brief This method is responsible for drawing the scene using forward rendering , and transparent objects
     */
    void drawForwardTransparencyMode(); 

    /**
     * @brief This method draw bounding boxes on the scene meshes.
     * !Note : must be used after every other mesh has been drawn , except the screen framebuffer , 
     * as BoundingBoxMesh uses the bound mesh's matrixes for it's own transformations.
     */
    void drawBoundingBoxes();

    /**
     * @brief Saves the light database pointer.
     * !This method does not clear nor free this object. 
     * @param database The pointer on the scene's lighting database. 
     */
    void setLightDatabasePointer(LightingDatabase* database){light_database = database;}

    /**
     * @brief Set the pointer on the camera used for the next render pass
     * 
     * @param _scene_camera 
     */
    void setCameraPointer(Camera* _scene_camera){scene_camera = _scene_camera;}

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
    LightingDatabase* light_database;
    Camera* scene_camera; 
};























#endif