#ifndef RENDERER_H
#define RENDERER_H

#include "constants.h"
#include "utils_3D.h"
#include "Loader.h"
#include "Mesh.h"
#include "Drawable.h" 
#include "TextureDatabase.h" 
#include "Camera.h"

/**
 * @file Renderer.h
 * Implementation of the renderer system 
 * 
 */


/**
 * @brief Renderer class definition
 * 
 */
class Renderer {
public:
	
	/**
	 * @brief Construct a new Renderer object
	 * 
	 */
	Renderer();
	
	/**
	 * @brief Destroy the Renderer object
	 * 
	 */
	virtual ~Renderer();	
	
	/**
	 * @brief Initialize the renderer's basic characteristics , like depth , background colors etc
	 * 
	 */
	void initialize();
	
	/**
	 * @brief Method setting up the meshes , and the scene camera
	 * 
	 * @return true If operation succeeded 
	 */
	bool prep_draw();
	
	/**
	 * @brief Initiates the draw calls for all objects in the scene
	 * 
	 */
	void draw(); 
	
	/**
	 * @brief Cleans up the former scene and replaces it with a new 
	 * 
	 * @param new_scene New scene to be rendererd
	 */
	void set_new_scene(std::vector<axomae::Mesh*> &new_scene);
	
	/**
	 * @brief Checks if all Drawable objects in the scene have been initialized
	 * 
	 * @return true If the scene is initialized and ready 
	 */
	bool scene_ready() ; 
	
	/**
	 * @brief Get the pointer to the MouseState structure 
	 * 
	 * @return MouseState* Pointer on the mouse_state variable , informations about the mouse positions , button clicked etc
	 * @see MouseState
	 * /
	MouseState* getMouseStatePointer(){ return &mouse_state;} ;  
	
	/**
	 * @brief Left click event behavior
	 * 
	 */
	void onLeftClick(); 
	
	/**
	 * @brief Right click event behavior
	 * 
	 */
	void onRightClick(); 
	
	/**
	 * @brief Left click release event behavior
	 * 
	 */
	void onLeftClickRelease(); 
	
	/**
	 * @brief Right click release event behavior
	 * 
	 */
	void onRightClickRelease();
	
	/**
	 * @brief Scroll down event behavior
	 * 
	 */
	void onScrollDown(); 
	
	/**
	 * @brief Scroll up event behavior
	 * 
	 */
	void onScrollUp(); 
	
	/**
	 * @brief Set the new screen dimensions 
	 * 
	 * @param width Width of the screen  
	 * @param height Height of the screen
	 */
	void setScreenSize(unsigned int width , unsigned int height);

public:
	std::vector<Drawable*> scene ; 				/**<The scene to be rendered*/ 
	bool start_draw ; 							/**<If the renderer is ready to draw*/
	TextureDatabase* texture_database; 			/**<Pointer on the texture database*/
	ShaderDatabase* shader_database; 			/**<Pointer on the shader database*/
	Camera *scene_camera ;						/**<Pointer on the scene camera*/
	MouseState mouse_state ;					/**<Pointer on the MouseState structure*/
	ScreenSize screen_size ; 					/**<Dimensions of the renderer windows*/
};

#endif
