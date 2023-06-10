#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H


#include "utils_3D.h" 


class BoundingBox {
public:
	
	/**
	 * @brief Construct a new Bounding Box object
	 * 
	 */
	BoundingBox();
	
	/**
	 * @brief Construct a new Bounding Box object
	 * 
	 * @param geometry 
	 */
	BoundingBox(const std::vector<float> &geometry);  
	
	/**
	 * @brief Destroy the Bounding Box object
	 * 
	 */
	virtual ~BoundingBox() ;

	/**
	 * @brief Get the Position of the AABB 
	 * 
	 * @return glm::vec3 Center of the AABB
	 */
	virtual glm::vec3 getPosition() const {return center; } 

	/**
	 * @brief Compute the position of the AABB in view space. 
	 * 
	 * @param modelview Modelview matrix : Model x View
	 * @return glm::vec4 Position of the AABB relative to the camera
	 */
	virtual glm::vec3 computeModelViewPosition(glm::mat4 modelview) const {return glm::vec3(modelview * glm::vec4(center , 1.f));}

private:
	glm::vec3 max_coords; 
	glm::vec3 min_coords;   
	glm::vec3 center ; 

}; 






#endif
