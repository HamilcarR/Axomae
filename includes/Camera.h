#ifndef CAMERA_H
#define CAMERA_H

#include "utils_3D.h" 


class Camera{
public:
	Camera(); 
	Camera(float radians , ScreenSize* screen ,  float clip_near , float clip_far , MouseState* pointer); 
	virtual ~Camera(); 
	virtual void computeViewSpace();
	virtual void computeProjectionSpace(); 
	virtual glm::mat4 getView(){return view; }
	virtual glm::mat4 getViewProjection(); 
	virtual void onLeftClick() = 0 ;  
	virtual void onRightClick() = 0 ;
	virtual void onLeftClickRelease() = 0 ; 
	virtual void onRightClickRelease() = 0 ; 
	virtual void movePosition() = 0 ; 


protected:
	float near ; 
	float far ;
	float fov ; 
	glm::mat4 projection ;	
	glm::mat4 view; 
	glm::vec3 position ; 
	glm::vec3 target ; 
	glm::vec3 right ;
	glm::vec3 direction;
	glm::vec3 camera_up ; 

	const glm::vec3 up ;
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

protected:
	virtual void rotate(); 
	virtual void movePosition() override ;


protected: 
	float angle ;
	float radius ; 
	glm::vec2 cursor_position ; 
	glm::vec3 start_position ; 
	glm::vec3 last_position ; 
	glm::quat rotation;
	glm::quat last_rotation ;
	glm::vec3 axis ;  

};




class FreePerspectiveCamera : public Camera{
public:
	FreePerspectiveCamera(); 
	FreePerspectiveCamera(float radians , ScreenSize* screen , float near , float far , float radius, MouseState* pointer); 
	virtual ~FreePerspectiveCamera(); 
	virtual void onLeftClick()  ;  
	virtual void onRightClick() ;
	virtual void onLeftClickRelease()  ; 
	virtual void onRightClickRelease() ; 
	virtual void movePosition() ; 
















};




#endif 
