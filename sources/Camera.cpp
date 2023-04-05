#include "../includes/Camera.h" 
#include <cmath> 

/**********************************************************************************************************************************************/
Camera::Camera() : up(glm::vec3(0,1,0)){
	position = glm::vec3(1,0,-1.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0);

}

Camera::Camera(float rad , ScreenSize *screen, float near , float far , MouseState* pointer) : up(glm::vec3(0,1,0)){
	position = glm::vec3(0,-100,-50.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0);
	view = glm::mat4(1.f);
	projection = glm::mat4(1.f);
	fov = rad ; 
	this->far = far ; 
	this->near = near ; 
	gl_widget_screen_size = screen ; 
	mouse_state_pointer = pointer ; 
}

Camera::~Camera(){

}

void Camera::computeProjectionSpace(){
	projection = glm::perspective(glm::radians(fov) , ((float) (gl_widget_screen_size->width)) / ((float) (gl_widget_screen_size->height)) , near , far); 	
}

glm::mat4 Camera::getViewProjection(){
	computeProjectionSpace(); 
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
/**********************************************************************************************************************************************/
ArcballCamera::ArcballCamera(){

}

ArcballCamera::ArcballCamera(float radians , ScreenSize* screen  , float near , float far , float radius , MouseState* pointer): Camera(radians ,screen, near , far , pointer) {
	angle = 0 ; 
	this->radius = radius ; 
	start_position =  position = glm::vec3(0 , 0 , radius)  ; 
	cursor_position = glm::vec2(0); 
	rotation = last_rotation = glm::quat(1.f , 0.f , 0.f , 0.f ) ; 
	axis = glm::vec3(0.f); 
}

ArcballCamera::~ArcballCamera(){


}


void ArcballCamera::computeViewSpace(){
	cursor_position = glm::vec2(mouse_state_pointer->pos_x , mouse_state_pointer->pos_y) ; 
	if(mouse_state_pointer -> left_button_clicked){
		movePosition(); 
		rotate(); 
		glm::mat4 rotation_matrix = glm::mat4_cast(rotation) ; 
		position = rotation_matrix * glm::vec4(position + glm::vec3(0 , 0 , radius) , 0) ; 
		direction = glm::normalize(position - target) ;
		right = glm::normalize(glm::cross(up , direction)); 
		camera_up =   glm::cross(direction , right); 
		view = glm::lookAt(position , target , camera_up);	
	}
}

void ArcballCamera::rotate(){
	axis = glm::normalize(glm::cross(start_position , position)); 
	angle = glm::acos(glm::dot(position , start_position)); 
	std::cout << angle << "\n" ; 
	rotation = glm::angleAxis(angle , axis) ; 
	rotation =  rotation * last_rotation; 
}


static float get_z_axis(float x , float y , float radius) {
	if( ((x * x) + (y * y)) <= (radius * radius / 2) )
		return (float) sqrt((radius * radius) - (x * x) - (y * y)) ; 
	else
		return (float) ((radius * radius)/2)/sqrt((x * x) + (y * y)) ; 
}

void ArcballCamera::movePosition(){
	if(mouse_state_pointer->left_button_clicked){
		position.x = ((cursor_position.x - (gl_widget_screen_size->width/2)) / (gl_widget_screen_size->width/2)) * radius; 
		position.y = (((gl_widget_screen_size->height/2) - cursor_position.y) / (gl_widget_screen_size->height/2)) * radius; 
		position.z = get_z_axis(position.x , position.y , radius) ;  	
		std::cout << position.x << ";" << position.y << ";" << position.z  << "\n" ; 
		position = glm::normalize(position); 
	}	
}

void ArcballCamera::onLeftClick(){
	start_position.x = ((cursor_position.x - (gl_widget_screen_size->width/2)) / (gl_widget_screen_size->width/2)) * radius; 
	start_position.y = (((gl_widget_screen_size->height/2) - cursor_position.y) / (gl_widget_screen_size->height/2)) * radius; 
	start_position.z = get_z_axis(start_position.x , start_position.y , radius) ; 
	start_position = glm::normalize(start_position); 
}

void ArcballCamera::onLeftClickRelease(){
	last_rotation =  rotation  ; 
	rotation = glm::quat(1.f , 0.f , 0.f , 0.f); 
	last_position = position ; 
}



void ArcballCamera::onRightClick(){
}


void ArcballCamera::onRightClickRelease(){
}



/**********************************************************************************************************************************************/

FreePerspectiveCamera::FreePerspectiveCamera(){

}

FreePerspectiveCamera::FreePerspectiveCamera(float radians , ScreenSize* screen , float near , float far , float radius , MouseState* pointer): Camera(radians , screen ,  near , far , pointer) {

}

FreePerspectiveCamera::~FreePerspectiveCamera(){


}
void FreePerspectiveCamera::movePosition(){

}

void FreePerspectiveCamera::onLeftClick(){
}

void FreePerspectiveCamera::onLeftClickRelease(){
}



void FreePerspectiveCamera::onRightClick(){
}


void FreePerspectiveCamera::onRightClickRelease(){
}















