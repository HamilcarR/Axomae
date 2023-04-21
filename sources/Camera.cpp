#include "../includes/Camera.h" 
#include <cmath> 

#define DELTA_ZOOM 1.f 
#define ANGLE_EPSILON 0.0001f
#define VECTOR_EPSILON 0.0001f 
/**********************************************************************************************************************************************/
Camera::Camera() : up(glm::vec3(0,1,0)){
	position = glm::vec3(1,0,-1.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0);
	type = EMPTY ; 
}

Camera::Camera(float rad , ScreenSize *screen, float near , float far , MouseState* pointer) : up(glm::vec3(0,1,0)){
	type = EMPTY ; 
	position = glm::vec3(0,0,-1.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0);
	view = glm::mat4(1.f);
	projection = glm::mat4(1.f);
	fov = rad ;
	view_projection = glm::mat4(1.f); 
	this->far = far ; 
	this->near = near ;
	gl_widget_screen_size = screen ; 
	mouse_state_pointer = pointer ; 
//	projection = glm::perspective(glm::radians(fov) , ((float) (gl_widget_screen_size->width)) / ((float) (gl_widget_screen_size->height)) , near , far); 	
	projection = glm::mat4(1.f); 
}

Camera::~Camera(){

}

glm::mat4 Camera::getProjection(){
	return projection; 
}
void Camera::reset(){
	projection = glm::mat4(1.f); 
	view = glm::mat4(1.f) ; 
	target = glm::vec3(0 , 0 , 0); 
	position = glm::vec3 (0 , 0 , -1.f); 
	direction = glm::vec3(0 , 0 , 0) ; 
	camera_up = glm::vec3(0 , 0 , 0) ; 
	view_projection = glm::mat4(1.f); 

}

void Camera::computeProjectionSpace(){
	projection = glm::perspective(glm::radians(fov) , ((float) (gl_widget_screen_size->width)) / ((float) (gl_widget_screen_size->height)) , near , far); 	
}

void Camera::computeViewProjection(){
	computeProjectionSpace(); 
	computeViewSpace(); 	
	view_projection =  projection * view ; 
}

glm::mat4 Camera::getViewProjection(){
	return view_projection  ; 
}

void Camera::computeViewSpace(){	
	direction = glm::normalize(position - target) ; 
	right = glm::normalize(glm::cross(up , direction)); 
	camera_up = glm::cross(direction , right); 
	view = glm::lookAt(position , glm::vec3(0) , camera_up);
}
/**********************************************************************************************************************************************/
ArcballCamera::ArcballCamera(){
	type = ARCBALL ;
	scene_rotation_matrix = glm::mat4(1.f); 
	angle = 0 ; 
	radius = 0 ;
	default_radius = radius ; 
	radius_updated = false ; 
	start_position =  position = glm::vec3(0 , 0 , radius)  ; 
	cursor_position = glm::vec2(0); 
	rotation = last_rotation = glm::quat(1.f , 0.f , 0.f , 0.f ) ; 
	axis = glm::vec3(0.f); 
}

ArcballCamera::ArcballCamera(float radians , ScreenSize* screen  , float near , float far , float radius , MouseState* pointer): Camera(radians ,screen, near , far , pointer) {
	type = ARCBALL ; 
	angle = 0 ; 
	this->radius = radius ;
	default_radius = radius ; 
	radius_updated = false ; 
	start_position =  position = glm::vec3(0 , 0 , radius)  ; 
	cursor_position = glm::vec2(0); 
	rotation = last_rotation = glm::quat(1.f , 0.f , 0.f , 0.f ) ; 
	axis = glm::vec3(0.f); 
	scene_rotation_matrix = glm::mat4(1.f); 
}

ArcballCamera::~ArcballCamera(){


}

void ArcballCamera::reset(){
	Camera::reset(); 
	angle = 0.f ;
	radius = default_radius ; 
	radius_updated = false ; 
	start_position = position = glm::vec3(0 , 0 , radius); 
	cursor_position = glm::vec2(0); 
	rotation = last_rotation = glm::quat(1.f , 0.f , 0.f , 0.f) ; 
	axis = glm::vec3(0.f); 
	

}

void ArcballCamera::computeViewSpace(){
	cursor_position = glm::vec2(mouse_state_pointer->pos_x , mouse_state_pointer->pos_y) ; 
	if(mouse_state_pointer -> left_button_clicked){
		movePosition(); 
		rotate(); 
		glm::mat4 rotation_matrix = glm::mat4_cast(rotation) ; 
		position = rotation_matrix * glm::vec4(position + glm::vec3(0 , 0 , radius) , 0) ; 
		view = glm::lookAt(glm::vec3(0 , 0 , radius) , target , up)  ; 
		scene_rotation_matrix = rotation_matrix ; 	
	}
	else{
		view = glm::lookAt(glm::vec3(0 , 0 , radius) , target , up) ; 
		scene_rotation_matrix = glm::mat4_cast(last_rotation); 
	}
}

void ArcballCamera::rotate(){
	axis = glm::normalize(glm::cross(start_position , position + glm::vec3(VECTOR_EPSILON))); 
	float temp_angle  = glm::acos(glm::dot(position , start_position)) ; 
	angle = temp_angle  ;  
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

void ArcballCamera::updateZoom(float step){
	radius += step ; 
}

void ArcballCamera::zoomIn(){
	if((radius - DELTA_ZOOM) >= DELTA_ZOOM)
		updateZoom(-DELTA_ZOOM); 
}

void ArcballCamera::zoomOut(){
	updateZoom(DELTA_ZOOM); 
}

/**********************************************************************************************************************************************/

FreePerspectiveCamera::FreePerspectiveCamera(){
	type = PERSPECTIVE ; 
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

void FreePerspectiveCamera::zoomIn(){

}
void FreePerspectiveCamera::zoomOut(){

}













