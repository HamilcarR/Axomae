#ifndef CAMERA_H
#define CAMERA_H

#include "utils_3D.h" 


class Camera{
public:
	enum TYPE : signed {EMPTY = -1 , ARCBALL = 0 , PERSPECTIVE = 1} ; 
	Camera(); 
	Camera(float radians , ScreenSize* screen ,  float clip_near , float clip_far , MouseState* pointer); 
	virtual ~Camera(); 
	virtual void computeViewSpace();
	virtual void computeProjectionSpace();
	virtual void computeViewProjection(); 
	virtual glm::mat4 getView(){return view; }
	virtual glm::mat4 getViewProjection(); 
	virtual glm::mat4 getProjection() ; 
	virtual void onLeftClick() = 0 ;  
	virtual void onRightClick() = 0 ;
	virtual void onLeftClickRelease() = 0 ; 
	virtual void onRightClickRelease() = 0 ; 
	virtual void movePosition() = 0 ; 
	virtual void zoomIn() = 0 ; 
	virtual void zoomOut() = 0 ; 
	virtual void reset() ;
	virtual const glm::mat4& getSceneRotationMatrix() const = 0;
	virtual const glm::mat4& getSceneTranslationMatrix() const = 0 ; 
	virtual const glm::mat4& getSceneModelMatrix() const = 0 ; 
	TYPE getType() const {return type ; } 
protected:
	TYPE type ; 
	float near ; 
	float far ;
	float fov ; 
	glm::mat4 projection ;	
	glm::mat4 view; 
	glm::mat4 view_projection ; 
	glm::vec3 position ; 
	glm::vec3 target ; 
	glm::vec3 right ;
	glm::vec3 direction;
	glm::vec3 camera_up ; 
	const glm::vec3 world_up ;
	MouseState *mouse_state_pointer ; // only a pointer , No free needed 
	ScreenSize *gl_widget_screen_size; // only a pointer . No free needed
};


class ArcballCamera : public Camera{
public:
	ArcballCamera(); 
	ArcballCamera(float radians , ScreenSize* screen , float near , float far , float radius, MouseState* pointer); 
	virtual ~ArcballCamera();
	virtual void computeViewSpace(); 
	virtual void onLeftClick() override ; 
	virtual void onRightClick() override ;
	virtual void onLeftClickRelease() override; 
	virtual void onRightClickRelease() override;
	virtual void zoomIn() override ; 
	virtual void zoomOut() override ;
	virtual void reset() ;
	virtual const glm::mat4& getSceneModelMatrix() const ; 
	virtual const glm::mat4& getSceneTranslationMatrix() const ;  
	virtual const glm::mat4& getSceneRotationMatrix() const ;  
protected:
	virtual void rotate();
	virtual void translate();
	virtual void movePosition() override ;
	virtual void updateZoom(float step) ; 

protected: 
	float angle ;
	float radius ; 
	glm::vec2 cursor_position ; 
	glm::vec3 start_position ; 
	glm::vec3 last_position ; 
	glm::quat rotation;
	glm::quat last_rotation ;
	glm::mat4 translation ; 
	glm::mat4 last_translation ; 
	glm::vec3 axis ; 
	glm::vec3 panning_offset ; 
	bool radius_updated ; 
private:
	glm::mat4 scene_rotation_matrix;
	glm::mat4 scene_translation_matrix;
	glm::mat4 scene_model_matrix ; 
	glm::vec3 prev_delta ;
	glm::vec3 scene_position ; 
	glm::vec3 delta_position ; 
	float default_radius; 
	

};




class FreePerspectiveCamera : public Camera{
public:
	FreePerspectiveCamera(); 
	FreePerspectiveCamera(float radians , ScreenSize* screen , float near , float far , float radius, MouseState* pointer); 
	virtual ~FreePerspectiveCamera(); 
	virtual void onLeftClick() override ;  
	virtual void onRightClick() override ;
	virtual void onLeftClickRelease() override ; 
	virtual void onRightClickRelease() override ; 
	virtual void movePosition() override ; 
	virtual void zoomIn() override ; 
	virtual void zoomOut() override ; 















};




#endif 
