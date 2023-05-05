#include "../includes/Camera.h" 
#include <cmath> 

constexpr float DELTA_ZOOM = 1.f ;  
constexpr float ANGLE_EPSILON = 0.0001f;  // we use these to avoid nan values when angles or vector lengths become too small 
constexpr float VECTOR_EPSILON = 0.0001f;  //
constexpr float PANNING_SENSITIVITY = 10.f;  








/**********************************************************************************************************************************************/
Camera::Camera() : world_up(glm::vec3(0,1,0)){
	position = glm::vec3(1,0,-1.f); 
	target = glm::vec3(0,0,0);
	direction = glm::vec3(0,0,0);
	right = glm::vec3(0,0,0);
	type = EMPTY ; 
}

Camera::Camera(float rad , ScreenSize *screen, float near , float far , MouseState* pointer) :world_up(glm::vec3(0,1,0)){
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
	right = glm::normalize(glm::cross(world_up , direction)); 
	camera_up = glm::cross(direction , right); 
	view = glm::lookAt(position , glm::vec3(0) , camera_up);
}
/**********************************************************************************************************************************************/
ArcballCamera::ArcballCamera(){
	reset(); 
}

ArcballCamera::ArcballCamera(float radians , ScreenSize* screen  , float near , float far , float radius , MouseState* pointer): Camera(radians ,screen, near , far , pointer) {
	reset() ; 	
	default_radius = radius ; 
	this->radius = radius ; 
}

ArcballCamera::~ArcballCamera(){


}

void ArcballCamera::reset(){
	Camera::reset();
	type = ARCBALL ; 
	angle = 0.f ;
	radius = default_radius ; 
	radius_updated = false ; 
	start_position = position = glm::vec3(0 , 0 , radius); 
	cursor_position = glm::vec2(0); 
	rotation = last_rotation = glm::quat(1.f , 0.f , 0.f , 0.f) ; 
	axis = glm::vec3(0.f); 
	target = glm::vec3(0.f) ;
	panning_offset = glm::vec3(0) ;
	delta_position = glm::vec3(0.f); 
	translation = last_translation = scene_translation_matrix = scene_rotation_matrix = scene_model_matrix = glm::mat4(1.f) ; 


}

void ArcballCamera::rotate(){
	axis = glm::normalize(glm::cross(start_position , position + glm::vec3(VECTOR_EPSILON))); 
	float temp_angle  = glm::acos(glm::dot(position , start_position)) ; 
	angle = temp_angle > ANGLE_EPSILON ? temp_angle : ANGLE_EPSILON; 
	rotation = glm::angleAxis(angle , axis) ;
	rotation =  rotation * last_rotation; 
}

	
void ArcballCamera::translate(){
	delta_position = position - start_position ;
	auto new_delta = glm::vec3(glm::inverse(scene_rotation_matrix) * (glm::vec4(delta_position , 1.f)))  ; 
	panning_offset +=  new_delta * PANNING_SENSITIVITY ;
	start_position = position ;	
}



void ArcballCamera::computeViewSpace(){
	cursor_position = glm::vec2(mouse_state_pointer->pos_x , mouse_state_pointer->pos_y) ; 
	if(mouse_state_pointer -> left_button_clicked){
		movePosition(); 
		rotate(); 
		glm::mat4 rotation_matrix = glm::mat4_cast(rotation) ; 
		position = rotation_matrix * glm::vec4(position + glm::vec3(0 , 0 , radius) , 0) ; 
		scene_rotation_matrix = rotation_matrix ; 	
	}
	else if(mouse_state_pointer -> right_button_clicked){
		movePosition(); 
		translate() ; 
		translation = glm::translate(glm::mat4(1.f)  , panning_offset ) ; 	
		scene_translation_matrix = translation ; 
		last_translation = translation; 
	}
	else{
		scene_rotation_matrix = glm::mat4_cast(last_rotation);
		scene_translation_matrix = last_translation ;
	}
		scene_model_matrix = scene_rotation_matrix * scene_translation_matrix; 	
		direction = target - glm::vec3(0 , 0 , radius) ; 
		view = glm::lookAt(glm::vec3(0 , 0 , radius) , target , world_up)  ; 
}

const glm::mat4& ArcballCamera::getSceneRotationMatrix() const {
	return scene_rotation_matrix ; 
}

const glm::mat4& ArcballCamera::getSceneTranslationMatrix() const {
	return scene_translation_matrix ; 
}

const glm::mat4& ArcballCamera::getSceneModelMatrix() const {
	return scene_model_matrix ; 
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
	if(mouse_state_pointer->right_button_clicked){
		position.x = ((cursor_position.x - (gl_widget_screen_size->width/2)) / (gl_widget_screen_size->width/2)) ; 
		position.y = (((gl_widget_screen_size->height/2) - cursor_position.y) / (gl_widget_screen_size->height/2)) ; 
		position.z = 0.f ; 
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
	start_position.x = ((cursor_position.x - (gl_widget_screen_size->width/2)) / (gl_widget_screen_size->width/2)) ; 
	start_position.y = (((gl_widget_screen_size->height/2) - cursor_position.y) / (gl_widget_screen_size->height/2)) ; 
	start_position.z = 0.f ; 
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













