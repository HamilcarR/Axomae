#include "../includes/Loader.h"
#include <fstream>
#include <unistd.h> 
#include <QImage>
#include <QBuffer>
#include <QByteArray>




namespace axomae{

Loader* Loader::instance = nullptr;

Loader::Loader(){}

Loader::~Loader(){}

Loader* Loader::getInstance(){
	if(instance == nullptr)
		instance = new Loader() ; 
	return instance ; 
}


/* copy texture data from GLB , to ARGB8888 buffer*/

static void copyTexels(TextureData *totexture , aiTexture *fromtexture){
	if(totexture != nullptr){
		if(fromtexture->mHeight != 0){ //checking if texture is uncompressed
			unsigned int width = 0 ; 
			unsigned int height = 0 ; 	
			totexture->width = width = fromtexture->mWidth ; 
			totexture->height = height = fromtexture->mHeight ; 
			totexture->data = new uint32_t[totexture->width * totexture->height] ; 
			for(unsigned int i = 0 ; i < width * height ; i ++){
				uint8_t a = fromtexture->pcData[i].a;
				uint8_t r = fromtexture->pcData[i].r; 
				uint8_t g = fromtexture->pcData[i].g; 
				uint8_t b = fromtexture->pcData[i].b; 
				uint32_t rgba = (a << 24) | (b << 16) | (g << 8) | r ; 
				totexture->data[i] = rgba ; 
			}	
		}
		else{
			QImage image ;
			uint8_t *buffer = new uint8_t[fromtexture->mWidth] ; 
			memcpy(buffer , fromtexture->pcData , fromtexture->mWidth); 
		 	image.loadFromData((const unsigned char*) buffer , fromtexture->mWidth ) ; 
			totexture->data = new uint32_t[image.width() * image.height()]; 
			memcpy((void*) totexture->data , (void*) image.bits() , image.width() * image.height() * sizeof(uint32_t)) ; 
			totexture->width = image.width() ; 
			totexture->height = image.height() ; 
			std::cout << "image of size " << totexture->width << " x " << totexture->height << " uncompressed " << std::endl ; 
		}
	}
}

static Material loadMaterial(const aiScene* scene , const aiMaterial* material){
	Material mesh_material; 
	TextureData diffuse , metallic , roughness ;
	diffuse.name = "diffuse" ; 
	metallic.name = "metallic" ; 
	roughness.name = "roughness" ; 
	unsigned int color_index = 0, metallic_index = 0 , roughness_index = 0; 
	aiString color_texture , normal_texture , metallic_texture , roughness_texture , occlusion_texture ; //we get indexes of embedded textures , since we will use GLB format  
	std::string color_index_string , metallic_index_string , roughness_index_string ; // returned index is in the form *X , we want to get rid of *
	material->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE , &color_texture) ; 
	material->GetTexture(AI_MATKEY_METALLIC_TEXTURE , &metallic_texture) ; 
	material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE , &roughness_texture) ;
	color_index_string = color_texture.C_Str(); 
	metallic_index_string = metallic_texture.C_Str() ; 
	roughness_texture = roughness_texture.C_Str() ;
	if(color_index_string.size() != 0){
		color_index_string = color_index_string.substr(1);
		color_index = std::stoi(color_index_string); 
		copyTexels(&diffuse , &*scene->mTextures[color_index]); 
		mesh_material.textures.diffuse = diffuse ; 
		diffuse.clean() ; 
	}
	if(metallic_index_string.size() != 0){
		metallic_index_string = metallic_index_string.substr(1) ; 
		metallic_index = std::stoi(metallic_index_string) ; 
		copyTexels(&metallic , &*scene->mTextures[metallic_index]); 
		mesh_material.textures.metallic = metallic ; 
		metallic.clean() ; 
	}
	if(roughness_index_string.size() != 0){
		roughness_index_string = roughness_index_string.substr(1) ; 
		roughness_index = std::stoi(roughness_index_string) ; 
		copyTexels(&roughness , &*scene->mTextures[roughness_index]); 
		mesh_material.textures.roughness = roughness ;
		roughness.clean() ; 
	}
	return mesh_material ; 
}

std::vector<Mesh> Loader::load(const char* file){
	Assimp::Importer importer ;
	const aiScene *modelScene = importer.ReadFile(file , aiProcess_CalcTangentSpace | aiProcess_Triangulate  | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs) ;
	if(modelScene != nullptr){
		std::vector<Mesh> objects; 
		const aiMaterial* ai_material ;
		for(unsigned int i = 0 ; i < modelScene->mNumMeshes ; i++){
			const aiMesh* mesh = modelScene->mMeshes[i] ; 
			const char* mesh_name = mesh->mName.C_Str(); 
			std::string name(mesh_name); 
			Object3D object ;
			assert(mesh->HasTextureCoords(0) ) ; 
			ai_material = modelScene->mMaterials[modelScene->mMeshes[i]->mMaterialIndex]; 
			Material mesh_material = loadMaterial(modelScene , ai_material);  	
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
			Mesh loaded_mesh ; 
			loaded_mesh.geometry = object ; 
			loaded_mesh.material = mesh_material ;
			loaded_mesh.name = name ; 
			objects.push_back(loaded_mesh);
		}
		return objects ; 
	}
	else{
		std::cout << "Problem loading scene" << std::endl ; 
		return std::vector<Mesh>() ; 	
	}
}

void Loader::close(){
	delete instance ; 
	instance = nullptr; 
}

}
