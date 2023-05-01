#include "../includes/Loader.h"
#include <fstream>
#include <iostream>
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


static void loadTexture(const aiScene* scene , Material *material ,TextureData &texture ,aiString texture_string ,Texture::TYPE type ){
	TextureDatabase* texture_database = TextureDatabase::getInstance(); 
	std::string texture_index_string = texture_string.C_Str();
	std::cout << "Texture type loaded : " << texture.name << " / GLB index is: " << texture_index_string <<  "\n" ; 
	 if(texture_index_string.size() != 0){
		texture_index_string = texture_index_string.substr(1) ; 
		unsigned int texture_index_int = stoi(texture_index_string);
		if(!texture_database->contains(texture_index_int)){
			copyTexels(&texture , &*scene->mTextures[texture_index_int]) ; 
			texture_database->addTexture(texture_index_int , &texture , type); 
			material->addTexture(texture_index_int , type);
			texture.clean(); 
		}
		else
			material->addTexture(texture_index_int , type); 
	 }
	 else
		 std::cout << "Loader can't load texture\n" ;  
}

//TODO : optimize in case different meshes use a same texture
static Material loadMaterial(const aiScene* scene , const aiMaterial* material){
	Material mesh_material; 
	TextureData diffuse , metallic , roughness , normal , ambiantocclusion , emissive , specular ;
	diffuse.name = "diffuse" ; 
	metallic.name = "metallic" ; 
	roughness.name = "roughness" ; 
	normal.name = "normal" ; 
	ambiantocclusion.name= "occlusion" ; 
	specular.name = "specular" ; 
	emissive.name = "emissive" ; 
	unsigned int color_index = 0, metallic_index = 0 , roughness_index = 0; 
	aiString color_texture , normal_texture , metallic_texture , roughness_texture , emissive_texture , specular_texture , occlusion_texture ; //we get indexes of embedded textures , since we will use GLB format  
	material->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE , &color_texture) ; 
	material->GetTexture(AI_MATKEY_METALLIC_TEXTURE , &metallic_texture) ; 
	material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE , &roughness_texture) ;	

	loadTexture(scene , &mesh_material , diffuse , color_texture , Texture::DIFFUSE); 
	loadTexture(scene , &mesh_material , metallic , metallic_texture , Texture::METALLIC); 
	loadTexture(scene , &mesh_material , roughness , roughness_texture , Texture::ROUGHNESS); 
	if(material->GetTextureCount(aiTextureType_NORMALS) > 0){
		material->GetTexture(aiTextureType_NORMALS , 0 , &normal_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , normal , normal_texture , Texture::NORMAL); 
	}	
	if(material->GetTextureCount(aiTextureType_LIGHTMAP) > 0){
		material->GetTexture(aiTextureType_LIGHTMAP , 0 , &occlusion_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , ambiantocclusion , occlusion_texture , Texture::AMBIANTOCCLUSION); 
	}
	if(material->GetTextureCount(aiTextureType_SHEEN) > 0){
		material-> GetTexture(aiTextureType_SHEEN , 0 , &specular_texture , nullptr , nullptr , nullptr , nullptr , nullptr);
		loadTexture(scene , &mesh_material , specular , specular_texture , Texture::SPECULAR); 
	}
	if(material->GetTextureCount(aiTextureType_EMISSIVE) > 0){
		material->GetTexture(aiTextureType_EMISSIVE , 0 , &emissive_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , emissive , emissive_texture , Texture::EMISSIVE); 
	}
	return mesh_material ; 
}









std::pair<unsigned int , std::vector<Mesh*>> Loader::loadObjects(const char* file){
	TextureDatabase *texture_database = TextureDatabase::getInstance() ; 	
	std::vector<Mesh*> objects; 
	std::string vertex_shader = loadShader("../shaders/simple.vert") ; 
	std::string fragment_shader = loadShader("../shaders/simple.frag"); 
	texture_database->clean(); 
	Assimp::Importer importer ;	
	const aiScene *modelScene = importer.ReadFile(file , aiProcess_CalcTangentSpace | aiProcess_Triangulate  | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs ) ;
	if(modelScene != nullptr){
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
			Mesh *loaded_mesh = new Mesh() ; 
			loaded_mesh->geometry = object ; 
			loaded_mesh->material = mesh_material ;
			loaded_mesh->name = name ; 
			loaded_mesh->shader_program.setShadersRawText(vertex_shader , fragment_shader) ; 
			objects.push_back(loaded_mesh);
		}
		return std::pair<unsigned int , std::vector<Mesh*>> (modelScene->mNumTextures , objects) ; 
	}	
	else{
		std::cout << "Problem loading scene" << std::endl ; 
		return std::pair<unsigned int , std::vector<Mesh*>> (0 , std::vector<Mesh*>()) ; 
	}


}



std::vector<Mesh*> Loader::load(const char* file){
	std::pair<unsigned int , std::vector<Mesh*>> scene = loadObjects(file); 
	Mesh* cube_map = generateCubeMap(scene.first , false) ; 	
	if(cube_map != nullptr)
		scene.second.push_back(cube_map); 
	return scene.second ; 


}

void Loader::close(){
	if(instance != nullptr)
		delete instance ; 
	instance = nullptr; 
}


std::string Loader::loadShader(const char* filename){
	std::ifstream stream(filename) ; 
	std::string buffer; 
	std::string shader_text ; 
	while(getline(stream , buffer))
		shader_text = shader_text + buffer + "\n" ;
	stream.close(); 
	return shader_text ; 

}



Mesh* Loader::generateCubeMap(unsigned int num_textures , bool is_glb){
	Mesh *cube_map = new CubeMapMesh();  
	std::string vertex_shader = loadShader("../shaders/cubemap.vert"); 
	std::string fragment_shader = loadShader("../shaders/cubemap.frag"); 
	cube_map->shader_program.setShadersRawText(vertex_shader , fragment_shader); 
	TextureDatabase* texture_database = TextureDatabase::getInstance(); 	
	TextureData cubemap ; 
	QString skybox_folder = "sky" ; 
	QString path = "../Ressources/Skybox_Textures/Sky/" ;
	auto format = "jpg"; 
	QImage left(":/"+skybox_folder+"/negx.jpg" , format); 		
	QImage bot(":/"+skybox_folder+"/negy.jpg" , format); 		
	QImage front(":/"+skybox_folder+"/negz.jpg" , format); 		
	QImage right(":/"+skybox_folder+"/posx.jpg" , format); 		
	QImage top(":/"+skybox_folder+"/posy.jpg" , format); 		
	QImage back(":/"+skybox_folder+"/posz.jpg" , format); 
	std::vector<QImage> array = { right , left , top , bot , back , front} ; 
	cubemap.width = left.width() ; 
	cubemap.height = left.height(); 	
	cubemap.data = new uint32_t [cubemap.width * cubemap.height * 6] ;
	unsigned int k = 0 ; 
	for(unsigned i = 0 ; i < array.size() ; i++)
		for(unsigned y = 0 ; y < array[i].height() ; y++){		
			QRgb *line = reinterpret_cast<QRgb*>(array[i].scanLine(y)) ; 
			for(unsigned x = 0 ; x < array[i].width() ; x++){
				QRgb rgba = line[x] ; 
				int r = qRed(rgba) ; 
				int g = qGreen(rgba) ; 
				int b = qBlue(rgba) ; 
				int a = qAlpha(rgba) ; 
				uint32_t rgba_final = (a << 24) | (b << 16) | (g << 8) | r ; 
				cubemap.data[k] = rgba_final ; 
				k++ ;		
			}
		}
	
	Material material ; 
	texture_database->addTexture(num_textures + 1 , &cubemap , Texture::CUBEMAP ) ; 
	material.addTexture(num_textures + 1 , Texture::CUBEMAP); 
	cube_map->material = material ; 	
	return cube_map; 		

}








}
