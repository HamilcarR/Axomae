#ifndef RENDERER_H
#define RENDERER_H

#include "constants.h"
#include "utils_3D.h"
#include "Loader.h"
#include "Mesh.h"
#include "Drawable.h" 
#include "ResourceDatabaseManager.h"
#include "Camera.h"
#include "LightingDatabase.h"
#include "CameraFrameBuffer.h"
#include "Scene.h"

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
	 * @brief Construct a new Renderer object
	 * 
	 * @param width Width of the window
	 * @param height Height of the window
	 */
	Renderer(unsigned width , unsigned height); 

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
	 */
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
	void onResize(unsigned int width , unsigned int height);

	/**
	 * @brief Set the Default Framebuffer ID 
	 * 
	 * @param id ID of the default framebuffer
	 */
	void setDefaultFrameBufferId(unsigned id){default_framebuffer_id = id ; }

	

public:
	Scene *scene ; 								/**<The scene to be rendered*/ 
	bool start_draw ; 							/**<If the renderer is ready to draw*/
	ResourceDatabaseManager *resource_database; /**<The main database containing a texture database , and a shader database*/
	LightingDatabase* light_database;			/**<Light database object*/ 
	Camera *scene_camera ;						/**<Pointer on the scene camera*/
	MouseState mouse_state ;					/**<Pointer on the MouseState structure*/
	ScreenSize screen_size ; 					/**<Dimensions of the renderer windows*/
	CameraFrameBuffer *camera_framebuffer ;		/**<Main framebuffer attached to the view*/ 
	unsigned int default_framebuffer_id ;		/**<In the case the GUI uses other contexts and other framebuffers , we use this variable to reset the rendering to the default framebuffer*/ 
};

#endif
