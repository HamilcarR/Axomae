#ifndef MESHLISTVIEW_H
#define MESHLISTVIEW_H

#include "constants.h"
#include "Mesh.h" 


#include <QListWidget>


class MeshListView : public QListWidget {
public:
	MeshListView(QWidget* parent=nullptr);
	virtual ~MeshListView();
	void setList(std::vector<axomae::Mesh> &meshes); 
	void remove();

private:
	std::vector<QListWidgetItem*> mesh_names_list ; 

};






#endif 
