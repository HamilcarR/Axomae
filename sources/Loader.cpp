#include "../includes/Loader.h"
#include <fstream>
#include <iostream>
#include <unistd.h> 
#include <QImage>
#include <QBuffer>
#include <QByteArray>


/**
 * @file Loader.cpp
 * Loader implementation 
 * 
 */

namespace axomae{

Loader* Loader::instance = nullptr;

Loader::Loader(){
}

Loader::~Loader(){}

Loader* Loader::getInstance(){
	if(instance == nullptr)
		instance = new Loader() ; 
	return instance ; 
}

/**
 * The function copies texture data from a GLB file to an ARGB8888 buffer.
 * 
 * @param totexture A pointer to a TextureData struct that will hold the copied texture data.
 * @param fromtexture The aiTexture object containing the texture data to be copied.
 */
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
			image = image.convertToFormat(QImage::Format_ARGB32); 	
			totexture->data = new uint32_t[image.width() * image.height()];
			uint8_t * pointer_to_bits = image.bits(); 
			for(unsigned i = 0 ; i < image.width() * image.height() * sizeof(uint32_t); i++)
				((uint8_t*) totexture->data)[i] = image.bits()[i] ; 
			totexture->width = image.width() ; 
			totexture->height = image.height() ; 
			std::cout << "image of size " << totexture->width << " x " << totexture->height << " uncompressed " << std::endl ; 
		}
	}
}


static void loadTextureDummy(Material* material , Texture::TYPE type){
	TextureDatabase* texture_database = TextureDatabase::getInstance();
	int index = texture_database->addTexture(nullptr , type , true , true); 	
	std::cout << "Loading dummy texture at index : " << index << std::endl;  
	material->addTexture(index , type); 
}

/**
 * This function loads a texture from an aiScene and adds it to a Material object.
 * 
 * @param scene A pointer to the aiScene object which contains the loaded 3D model data.
 * @param material A pointer to a Material object that the texture will be added to.
 * @param texture The variable that stores the loaded texture data.
 * @param texture_string The name or index of the texture file to be loaded, stored as an aiString.
 * @param type The type of texture being loaded, which is of the enum type Texture::TYPE.
 */
static void loadTexture(const aiScene* scene , Material *material ,TextureData &texture ,aiString texture_string ,Texture::TYPE type ){
	TextureDatabase* texture_database = TextureDatabase::getInstance(); 
	std::string texture_index_string = texture_string.C_Str();
	std::cout << "Texture type loaded : " << texture.name << " / GLB index is: " << texture_index_string <<  "\n" ; 
	 if(texture_index_string.size() != 0){
		texture_index_string = texture_index_string.substr(1) ; 
		unsigned int texture_index_int = stoi(texture_index_string);  
		if(!texture_database->contains(texture_index_int)){
			copyTexels(&texture , &*scene->mTextures[texture_index_int]) ; 
			int index = texture_database->addTexture(&texture , type , false); 
			material->addTexture(index , type);
			texture.clean(); 
		}
		else
			material->addTexture(texture_index_int , type); 
	 }
	 else
		 std::cout << "Loader can't load texture\n" ;  
}



/**
 * The function loads textures for a given material in a 3D model scene.
 * 
 * @param scene a pointer to the aiScene object which contains the loaded 3D model data.
 * @param material The aiMaterial object that contains information about the material properties of a
 * 3D model.
 * 
 * @return a Material object.
 */

static Material loadAllTextures(const aiScene* scene , const aiMaterial* material ){
	Material mesh_material ; 	
	std::vector<Texture::TYPE> dummy_textures_type; 
	TextureData diffuse , metallic , roughness , normal , ambiantocclusion , emissive , specular , opacity;
	diffuse.name = "diffuse" ; 
	metallic.name = "metallic" ; 
	roughness.name = "roughness" ;
	opacity.name = "opacity" ;  
	normal.name = "normal" ; 
	ambiantocclusion.name= "occlusion" ; 
	specular.name = "specular" ; 
	emissive.name = "emissive" ; 
	unsigned int color_index = 0, metallic_index = 0 , roughness_index = 0; 
	aiString color_texture , opacity_texture ,  normal_texture , metallic_texture , roughness_texture , emissive_texture , specular_texture , occlusion_texture ; //we get indexes of embedded textures , since we will use GLB format  

	if(material->GetTextureCount(aiTextureType_BASE_COLOR) > 0){
		material->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE , &color_texture) ;
		loadTexture(scene , &mesh_material , diffuse , color_texture , Texture::DIFFUSE); 
	}
	else
		dummy_textures_type.push_back(Texture::DIFFUSE);

	if(material->GetTextureCount(aiTextureType_OPACITY) > 0){
		mesh_material.setTransparency(true) ;
		material->GetTexture(aiTextureType_OPACITY , 0 ,  &opacity_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , opacity , opacity_texture , Texture::OPACITY); 
	}
	else
		dummy_textures_type.push_back(Texture::OPACITY); 
	
	if(material->GetTextureCount(aiTextureType_METALNESS) > 0){
		material->GetTexture(AI_MATKEY_METALLIC_TEXTURE , &metallic_texture) ; 
		loadTexture(scene , &mesh_material , metallic , metallic_texture , Texture::METALLIC); 
	}
	else
		dummy_textures_type.push_back(Texture::METALLIC); 
	
	if(material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0){
		material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE , &roughness_texture) ;	
		loadTexture(scene , &mesh_material , roughness , roughness_texture , Texture::ROUGHNESS);
	}
	else
		dummy_textures_type.push_back(Texture::ROUGHNESS); 
	
	if(material->GetTextureCount(aiTextureType_NORMALS) > 0){
		material->GetTexture(aiTextureType_NORMALS , 0 , &normal_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , normal , normal_texture , Texture::NORMAL); 
	}
	else
		dummy_textures_type.push_back(Texture::NORMAL); 
	
	if(material->GetTextureCount(aiTextureType_LIGHTMAP) > 0){
		material->GetTexture(aiTextureType_LIGHTMAP , 0 , &occlusion_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , ambiantocclusion , occlusion_texture , Texture::AMBIANTOCCLUSION); 
	}
	else
		dummy_textures_type.push_back(Texture::AMBIANTOCCLUSION); 
	
	if(material->GetTextureCount(aiTextureType_SHEEN) > 0){
		material->GetTexture(aiTextureType_SHEEN , 0 , &specular_texture , nullptr , nullptr , nullptr , nullptr , nullptr);
		loadTexture(scene , &mesh_material , specular , specular_texture , Texture::SPECULAR); 
	}
	else
		dummy_textures_type.push_back(Texture::SPECULAR); 	
	
	if(material->GetTextureCount(aiTextureType_EMISSIVE) > 0){
		material->GetTexture(aiTextureType_EMISSIVE , 0 , &emissive_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , emissive , emissive_texture , Texture::EMISSIVE); 
	}
	else
		dummy_textures_type.push_back(Texture::EMISSIVE);
	
	for(auto it = dummy_textures_type.begin(); it != dummy_textures_type.end() ; it ++)
		loadTextureDummy(&mesh_material , *it);
	
	return mesh_material ; 
}



static float loadTransparencyValue(const aiMaterial* material){
	float transparency = 1.f ;
	float opacity = 1.f;  
	aiColor4D col ; 
	if (material->Get(AI_MATKEY_COLOR_TRANSPARENT, col) == AI_SUCCESS)
		transparency = col.a ; 
	else
		if (material->Get(AI_MATKEY_OPACITY , opacity) == AI_SUCCESS)
			transparency = 1.f - opacity ; 
	return transparency ; 
}


static Material loadMaterials(const aiScene* scene , const aiMaterial* material){
	Material mesh_material = loadAllTextures(scene , material);
	float transparency_factor = loadTransparencyValue(material);  	
	mesh_material.setTransparency(transparency_factor);
	return mesh_material ; 
}


/**
 * The function loads shader files and adds them to a shader database.
 */
void Loader::loadShaderDatabase(){	
	ShaderDatabase* shader_database = ShaderDatabase::getInstance(); 
	std::string vertex_shader = loadShader("../shaders/phong.vert") ; 
	std::string fragment_shader = loadShader("../shaders/phong.frag"); 
	std::string vertex_shader_cubemap = loadShader("../shaders/cubemap.vert"); 
	std::string fragment_shader_cubemap = loadShader("../shaders/cubemap.frag");
	std::string vertex_shader_screen_fbo = loadShader("../shaders/screen_fbo.vert"); 
	std::string fragment_shader_screen_fbo = loadShader("../shaders/screen_fbo.frag");  
	shader_database->addShader(vertex_shader , fragment_shader , Shader::BLINN) ; 
	shader_database->addShader(vertex_shader_cubemap , fragment_shader_cubemap , Shader::CUBEMAP) ; 
	shader_database->addShader(vertex_shader_screen_fbo , fragment_shader_screen_fbo , Shader::SCREEN_FRAMEBUFFER); 
}

/**
 * The function loads GLB objects and returns a pair containing the
 * number of textures and a vector of Mesh objects.
 * 
 * @param file The file path of the 3D model to be loaded.
 * 
 * @return A std::pair containing an unsigned int and a vector of Mesh pointers. The unsigned int
 * represents the number of textures in the loaded scene, and the vector contains pointers to the
 * loaded Mesh objects.
 */
std::pair<unsigned int , std::vector<Mesh*>> Loader::loadObjects(const char* file){
	TextureDatabase *texture_database = TextureDatabase::getInstance() ; 	
	ShaderDatabase *shader_database = ShaderDatabase::getInstance(); 
	std::vector<Mesh*> objects; 
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
			Material mesh_material = loadMaterials(modelScene , ai_material);  	
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
			Shader* shader_program = shader_database->get(Shader::BLINN) ;  //TODO : change for PBR and other nice shaders when needed 		
			Mesh *loaded_mesh = new Mesh(std::string(mesh->mName.C_Str()) , object , mesh_material , shader_program) ; //TODO : change shader_program with pointer to pair<shader::type , shader*> 	
			objects.push_back(loaded_mesh);
		}
		return std::pair<unsigned int , std::vector<Mesh*>> (modelScene->mNumTextures , objects) ; 
	}	
	else{
		std::cout << "Problem loading scene" << std::endl ; 
		return std::pair<unsigned int , std::vector<Mesh*>> (0 , std::vector<Mesh*>()) ; 
	}
}



/**
 * The function loads a file and generates a vector of meshes, including a cube map if applicable.
 * 
 * @param file A pointer to a character array representing the file path of the 3D model to be loaded.
 * 
 * @return A vector of Mesh pointers.
 */
std::vector<Mesh*> Loader::load(const char* file){
	TextureDatabase *texture_database = TextureDatabase::getInstance(); 	
	ShaderDatabase *shader_database = ShaderDatabase::getInstance(); 
	texture_database->softCleanse();
	shader_database->clean(); 
	loadShaderDatabase();  
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

/**
 * The function loads a shader from a file and returns it as a string.
 * 
 * @param filename String representing the name of the file to be loaded.
 * 
 * @return The function `loadShader` returns a `std::string` which contains the text read from the file
 * specified by the `filename` parameter.
 */

std::string Loader::loadShader(const char* filename){
	std::ifstream stream(filename) ; 
	std::string buffer; 
	std::string shader_text ; 
	while(getline(stream , buffer))
		shader_text = shader_text + buffer + "\n" ;
	stream.close(); 
	return shader_text ; 

}

/**
 * This function generates a cube map mesh with textures loaded from a specified folder.
 * 
 * @param num_textures The number of textures currently loaded in the texture database before adding
 * the cubemap texture.
 * @param is_glb The parameter "is_glb" is not used in the function and therefore has no effect on the
 * generated cube map.
 * 
 * @return A pointer to a Mesh object representing a cube map.
 */
Mesh* Loader::generateCubeMap(unsigned int num_textures , bool is_glb){
	Mesh *cube_map = new CubeMapMesh(); 
	ShaderDatabase *shader_database = ShaderDatabase::getInstance(); 
	TextureDatabase* texture_database = TextureDatabase::getInstance(); 	
	TextureData cubemap ; 
	QString skybox_folder = "castle" ;
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
	unsigned index = texture_database->addTexture(&cubemap , Texture::CUBEMAP , false) ; 
	material.addTexture(index , Texture::CUBEMAP);
	cube_map->material = material ; 
	cube_map->setShader(shader_database->get(Shader::CUBEMAP)) ; 
	cubemap.clean(); 	
	return cube_map; 		
}








}
