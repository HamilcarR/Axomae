#ifndef SCENESELECTOR_H
#define SCENESELECTOR_H

#include "Loader.h" 
#include "Mesh.h"

/**
 * @brief Keeps track of currently displayed mesh , and the parent 3D object
 */

class SceneSelector{
public:
	static SceneSelector* getInstance(); 
	static void remove() ;
	void setScene(std::vector<axomae::Mesh*> &meshes); 
	void toNext() ; 
	void toPrevious() ;
	axomae::Mesh* getCurrent() ; 


private:
	SceneSelector();
	virtual ~SceneSelector();	
	std::vector<axomae::Mesh*> scene ; 
	unsigned int mesh_index ;
	static SceneSelector *instance;
};




#endif 
