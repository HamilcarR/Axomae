#ifndef CAMERA_H
#define CAMERA_H

#include "utils_3D.h" 


class Camera{
public:
	Camera(); 
	Camera(float radians , float ratio , float clip_near , float clip_far); 
	virtual ~Camera(); 
	virtual void computeViewSpace();
	virtual glm::mat4 getView(){return view; }
	virtual glm::mat4 getViewProjection(); 
protected:
	glm::mat4 projection ; 
	glm::vec3 position ; 
	glm::vec3 target ; 
	glm::vec3 right ;
	glm::vec3 direction;
	glm::vec3 camera_up ; 
	glm::mat4 view; 
	const glm::vec3 up ; 
	
};




#endif 
