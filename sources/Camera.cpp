#include "../includes/Camera.h" 


Camera::Camera() : up(glm::vec3(0,1,0)){
	position = glm::vec3(1,0,-1.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0); 
}

Camera::Camera(float rad , float ratio , float near , float far) : up(glm::vec3(0,1,0)){
	position = glm::vec3(0,-100,-50.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0);
	projection = glm::perspective(glm::radians(rad) , ratio , near , far); 
}

Camera::~Camera(){

}


glm::mat4 Camera::getViewProjection(){
	computeViewSpace(); 
	glm::mat4 result =  projection * getView() ; 
	return result ; 
}

void Camera::computeViewSpace(){
	direction = glm::normalize(position - target) ; 
	right = glm::normalize(glm::cross(up , direction)); 
	camera_up = glm::cross(direction , right); 
	view = glm::lookAt(position , glm::vec3(0) , camera_up);


}

