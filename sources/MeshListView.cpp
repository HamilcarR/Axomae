#include "../includes/MeshListView.h"


using namespace axomae;


MeshListView::MeshListView(QWidget* parent):QListWidget(parent){
	

}

MeshListView::~MeshListView(){	
	remove();  
}



void MeshListView::remove(){
	for(auto A : mesh_names_list){
		delete A ; 	
	}
	mesh_names_list.clear();  

}

void MeshListView::setList(std::vector<Mesh> &meshes){
	remove(); 
	for(Mesh A : meshes)
		mesh_names_list.push_back(new QListWidgetItem(QString(A.name.c_str()) , this , 0)); 
}
