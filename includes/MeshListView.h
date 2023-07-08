#ifndef MESHLISTVIEW_H
#define MESHLISTVIEW_H

#include "constants.h"
#include "Mesh.h" 
#include <QListWidget>

/**
 * @file MeshListView.h
 * This file implements the list of meshes in the UV renderer UI 
 * 
 */


/**
 * @class MeshListView
 * 
 */
class MeshListView : public QListWidget {
public:
	MeshListView(QWidget* parent=nullptr);
	virtual ~MeshListView();
	void setList(const std::vector<Mesh*> &meshes); 
	void remove();

private:
	std::vector<QListWidgetItem*> mesh_names_list ; 

};






#endif 
