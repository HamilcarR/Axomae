#include "../includes/Loader.h"



namespace axomae{

Loader* Loader::instance = nullptr;

Loader::Loader(){


}


Loader::~Loader(){



}


	
Loader* Loader::getInstance(){
	if(instance == nullptr)
		instance = new Loader() ; 
	return instance ; 

}


std::vector<Object3D> Loader::load(const char* file){
	Assimp::Importer importer ;
	const aiScene *modelScene = importer.ReadFile(file ,   aiProcess_CalcTangentSpace | aiProcess_Triangulate  | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs) ;
	if(modelScene != nullptr){
		std::vector<Object3D> objects; 
		for(unsigned int i = 0 ; i < modelScene->mNumMeshes ; i++){
			const aiMesh* mesh = modelScene->mMeshes[i] ; 
			Object3D object ;
			assert(mesh->HasTextureCoords(0) ) ; 
			for(unsigned int f = 0 ; f < mesh->mNumVertices ; f++){
				const aiVector3D* vert = &(mesh->mVertices[f]) ; 
				const aiVector3D* norm = &(mesh->mNormals[f]) ; 
				const aiVector3D* tex = &(mesh->mTextureCoords[0][f]) ; 
				const aiVector3D* tang = &(mesh->mTangents[f]); 
				const aiVector3D* bitang = &(mesh->mBitangents[f]) ; 
				object.vertices.push_back(vert->x); 
				object.vertices.push_back(vert->y); 
				object.vertices.push_back(vert->z); 
				object.uv.push_back(tex->x) ; 
				object.uv.push_back(tex->y) ; 
				object.normals.push_back(norm->x) ; 
				object.normals.push_back(norm->y) ; 
				object.normals.push_back(norm->z) ; 
				object.tangents.push_back(tang->x); 	
				object.tangents.push_back(tang->y); 	
				object.tangents.push_back(tang->z);
				object.bitangents.push_back(bitang->x) ; 
				object.bitangents.push_back(bitang->y) ; 
				object.bitangents.push_back(bitang->z) ; 
			}
			for(unsigned int ind = 0 ; ind < mesh->mNumFaces ; ind++){
				assert(mesh->mFaces[ind].mNumIndices==3) ;  
				object.indices.push_back(static_cast<unsigned int> (mesh->mFaces[ind].mIndices[0]));
				object.indices.push_back(static_cast<unsigned int> (mesh->mFaces[ind].mIndices[1]));
				object.indices.push_back(static_cast<unsigned int> (mesh->mFaces[ind].mIndices[2]));
			}
			std::cout << "object loaded : " << mesh->mName.C_Str()<< "\n" ; 
			objects.push_back(object) ;
			}
		return objects ; 
	}
	else{
		std::cout << "Problem loading scene" << std::endl ; 
		return std::vector<Object3D>() ; 	
	}
}


void Loader::close(){
	delete instance ; 
	instance = nullptr; 
}

}
