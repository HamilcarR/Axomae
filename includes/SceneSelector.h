#ifndef SCENESELECTOR_H
#define SCENESELECTOR_H

#include "Loader.h" 
#include "Mesh.h"

/**
 * @brief Keeps track of currently displayed mesh , and the parent 3D object
 */

class SceneSelector{
public:
	SceneSelector(std::vector<axomae::Mesh> &meshes);
	virtual ~SceneSelector();
	axomae::Mesh getNext() ; 
	axomae::Mesh getPrevious() ;
	axomae::Mesh getCurrent() ; 


private:
	std::vector<axomae::Mesh> scene ; 
	unsigned int mesh_index ;
};




#endif 
