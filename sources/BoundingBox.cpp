#include "../includes/BoundingBox.h" 

using namespace axomae ; 

glm::vec3 calculateCenter(glm::vec3 min_coords , glm::vec3 max_coords);

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
    center = glm::vec3(0.f); 
    max_coords = glm::vec3(0.f); 
    min_coords = glm::vec3(0.f); 
}

//TODO: [AX-12] Parallelize bounding box computation
BoundingBox::BoundingBox(const std::vector<float> &vertices):BoundingBox(){
    center = glm::vec3(0 , 0 , 0); 
    max_coords = glm::vec3(-INT_MAX); 
    min_coords = glm::vec3(INT_MAX); 
    for(unsigned i = 0 ; i < vertices.size(); i+=3){        
        glm::vec3 compare = glm::vec3(vertices[i] , vertices[i+1] , vertices[i+2]);  
        max_coords = update_max(max_coords , compare);
        min_coords = update_min(min_coords , compare);  
    }
    center = calculateCenter(min_coords , max_coords);  
}

BoundingBox::BoundingBox(glm::vec3 _min_coords , glm::vec3 _max_coords){
    center = calculateCenter(_min_coords , _max_coords); 
    min_coords = _min_coords; 
    max_coords = _max_coords; 
}

BoundingBox::~BoundingBox(){
}

BoundingBox operator*(const glm::mat4& matrix , const BoundingBox& bounding_box){
    glm::vec3 min_c = matrix * glm::vec4(bounding_box.getMinCoords() , 1.f);
    glm::vec3 max_c = matrix * glm::vec4(bounding_box.getMinCoords() , 1.f); 
    return BoundingBox(min_c , max_c);  
}

std::pair<std::vector<float> , std::vector<unsigned>> BoundingBox::getVertexArray() const {
    std::vector<float> vertices = {
        min_coords.x , min_coords.y , max_coords.z , //0
        max_coords.x , min_coords.y , max_coords.z , //1 
        max_coords.x , max_coords.y , max_coords.z , //2 
        min_coords.x , max_coords.y , max_coords.z , //3 
        max_coords.x , min_coords.y , min_coords.z , //4
        max_coords.x , max_coords.y , min_coords.z , //5 
        min_coords.x , max_coords.y , min_coords.z , //6
        min_coords.x , min_coords.y , min_coords.z   //7
    };
    std::vector<unsigned> indices = {
        0 , 1 , 2 , 
        0 , 2 , 3 , 
        1 , 4 , 5 , 
        1 , 5 , 2 , 
        7 , 6 , 5 , 
        7 , 5 , 4 , 
        3 , 6 , 7 , 
        3 , 7 , 0 , 
        2 , 5 , 6 , 
        2 , 6 , 3 , 
        7 , 4 , 1 , 
        7 , 1 , 0
    };
    return std::pair<std::vector<float> , std::vector<unsigned>>(vertices , indices);  
}


glm::vec3 calculateCenter(glm::vec3 min_coords , glm::vec3 max_coords){
    glm::vec3 center ; 
    center.x = (max_coords.x + min_coords.x) / 2 ; 
    center.y = (max_coords.y + min_coords.y) / 2 ; 
    center.z = (max_coords.z + min_coords.z) / 2 ; 
    return center; 
}