#include "../includes/BoundingBox.h" 

using namespace axomae ; 


glm::vec3 update_min(glm::vec3 min_vec, glm::vec3 compared){
    if(compared.x <= min_vec.x)
        min_vec.x = compared.x ; 
    if(compared.y <= min_vec.y)
        min_vec.y = compared.y ; 
    if(compared.z <= min_vec.z)
        min_vec.z = compared.z ;
    return min_vec ;  
}

glm::vec3 update_max(glm::vec3 max_vec, glm::vec3 compared){
    if(compared.x > max_vec.x)
        max_vec.x = compared.x ; 
    if(compared.y > max_vec.y)
        max_vec.y = compared.y ; 
    if(compared.z > max_vec.z)
        max_vec.z = compared.z ;
    return max_vec ;  
}


BoundingBox::BoundingBox(){
}

BoundingBox::BoundingBox(const std::vector<float> &vertices){
    center = glm::vec3(0 , 0 , 0); 
    max_coords = glm::vec3(-INT_MAX); 
    min_coords = glm::vec3(INT_MAX); 
    for(unsigned i = 0 ; i < vertices.size(); i+=3){
        glm::vec3 compare = glm::vec3(vertices[i] , vertices[i+1] , vertices[i+2]);  
        max_coords = update_max(max_coords , compare);
        min_coords = update_min(min_coords , compare);  
    }
    center.x = (max_coords.x + min_coords.x) / 2 ; 
    center.y = (max_coords.y + min_coords.y) / 2 ; 
    center.z = (max_coords.z + min_coords.z) / 2 ; 
}


BoundingBox::~BoundingBox(){
}
