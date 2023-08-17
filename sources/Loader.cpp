#include <fstream>
#include <iostream>
#include <unistd.h> 
#include <QImage>
#include <QBuffer>
#include <QByteArray>
#include "../includes/Loader.h"
#include "../includes/PerformanceLogger.h"
#include "../includes/Mutex.h"
#include "../includes/SceneNodeBuilder.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb/stb_image.h"
/**
 * @file Loader.cpp
 * Loader implementation 
 * 
 */

namespace axomae{
Loader::Loader(){
	resource_database = ResourceDatabaseManager::getInstance();
}
Loader::~Loader(){}


Mutex mutex; 
constexpr unsigned int THREAD_POOL_SIZE = 8 ; 
void thread_copy_buffer(unsigned int start_index , unsigned int end_index , uint8_t* from , uint8_t* dest){
	for(unsigned i = start_index ; i < end_index ; i++)
		dest[i] = from[i] ; 
}

void async_copy_buffer(unsigned int width , unsigned int height , uint8_t* from , uint8_t* dest){
	size_t total_buffer_size = width * height * sizeof(uint32_t); 
	size_t thread_buffer_size = total_buffer_size / THREAD_POOL_SIZE ; //TODO : implement the case where total_buffer_sze < THREAD_POOL_SIZE 
	uint8_t work_remainder = total_buffer_size % THREAD_POOL_SIZE ; 
	size_t last_thread_job_size = thread_buffer_size + work_remainder ; 
	size_t range_min = 0 ; 
	size_t range_max = 0 ;
	std::vector<std::future<void>> future_results ; 
	unsigned i = 0 ;  
	for(i ; i < THREAD_POOL_SIZE  ; i++){
		range_min = i * thread_buffer_size; 
		range_max = i * thread_buffer_size + thread_buffer_size ; 
		if(i == THREAD_POOL_SIZE - 1)
			range_max = i * thread_buffer_size + last_thread_job_size ; 
		future_results.push_back(
			std::async(thread_copy_buffer , range_min , range_max , from , dest)
		);	
	}
	for(std::vector<std::future<void>>::iterator it = future_results.begin() ; it != future_results.end() ; it++)
		it->get(); 
}

/**
 * The function copies texture data from a GLB file to an ARGB8888 buffer.
 * 
 * @param totexture A pointer to a TextureData struct that will hold the copied texture data.
 * @param fromtexture The aiTexture object containing the texture data to be copied.
 */
static void copyTexels(TextureData *totexture , aiTexture *fromtexture){ 
	if(totexture != nullptr){
		/* If mHeight != 0 , the texture is uncompressed , and we read it as is */
		if(fromtexture->mHeight != 0){
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
		/* If mHeight = 0 , the texture is compressed , and we need to uncompress and convert it to ARGB32 */
		else{
			QImage image ;
			uint8_t *buffer = new uint8_t[fromtexture->mWidth] ; 
			memcpy(buffer , fromtexture->pcData , fromtexture->mWidth);
			image.loadFromData((const unsigned char*) buffer , fromtexture->mWidth ) ;
			image = image.convertToFormat(QImage::Format_ARGB32); 	
			unsigned image_width = image.width(); 
			unsigned image_height = image.height();
			totexture->data = new uint32_t[image_width * image_height];
			memset(totexture->data , 0 , image_width * image_height * sizeof(uint32_t));
			uint8_t* dest_buffer = (uint8_t*) totexture->data ; 
			uint8_t* from_buffer = image.bits() ;
			async_copy_buffer(image_width , image_height , from_buffer , dest_buffer); 
			totexture->width = image_width ; 
			totexture->height = image_height ; 
			std::cout << "image of size " << totexture->width << " x " << totexture->height << " uncompressed " << std::endl ; 
		}
	}
}


static void loadTextureDummy(Material* material , Texture::TYPE type , TextureDatabase* texture_database){
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
static void loadTexture(const aiScene* scene , Material *material ,TextureData &texture ,aiString texture_string ,Texture::TYPE type , TextureDatabase* texture_database){
	std::string texture_index_string = texture_string.C_Str();
	std::cout << "Texture type loaded : " << texture.name << " / GLB index is: " << texture_index_string <<  "\n" ; 
	if(texture_index_string.size() != 0){
		texture_index_string = texture_index_string.substr(1) ; 
		unsigned int texture_index_int = stoi(texture_index_string);  
		if(!texture_database->contains(texture_index_int)){
			copyTexels(&texture , scene->mTextures[texture_index_int]) ;
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
static Material loadAllTextures(const aiScene* scene , const aiMaterial* material , TextureDatabase* texture_database){
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
		loadTexture(scene , &mesh_material , diffuse , color_texture , Texture::DIFFUSE , texture_database); 
	}
	else
		dummy_textures_type.push_back(Texture::DIFFUSE);

	if(material->GetTextureCount(aiTextureType_OPACITY) > 0){
		mesh_material.setTransparency(true) ;
		material->GetTexture(aiTextureType_OPACITY , 0 ,  &opacity_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , opacity , opacity_texture , Texture::OPACITY , texture_database ); 
	}
	else
		dummy_textures_type.push_back(Texture::OPACITY); 
	
	if(material->GetTextureCount(aiTextureType_METALNESS) > 0){
		material->GetTexture(AI_MATKEY_METALLIC_TEXTURE , &metallic_texture) ; 
		loadTexture(scene , &mesh_material , metallic , metallic_texture , Texture::METALLIC , texture_database); 
	}
	else
		dummy_textures_type.push_back(Texture::METALLIC); 
	
	if(material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0){
		material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE , &roughness_texture) ;	
		loadTexture(scene , &mesh_material , roughness , roughness_texture , Texture::ROUGHNESS , texture_database);
	}
	else
		dummy_textures_type.push_back(Texture::ROUGHNESS); 
	
	if(material->GetTextureCount(aiTextureType_NORMALS) > 0){
		material->GetTexture(aiTextureType_NORMALS , 0 , &normal_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , normal , normal_texture , Texture::NORMAL , texture_database); 
	}
	else
		dummy_textures_type.push_back(Texture::NORMAL); 
	
	if(material->GetTextureCount(aiTextureType_LIGHTMAP) > 0){
		material->GetTexture(aiTextureType_LIGHTMAP , 0 , &occlusion_texture , nullptr , nullptr , nullptr , nullptr , nullptr); 
		loadTexture(scene , &mesh_material , ambiantocclusion , occlusion_texture , Texture::AMBIANTOCCLUSION , texture_database); 
	}
	else
		dummy_textures_type.push_back(Texture::AMBIANTOCCLUSION); 
	
	if(material->GetTextureCount(aiTextureType_SHEEN) > 0){
		material->GetTexture(aiTextureType_SHEEN , 0 , &specular_texture , nullptr , nullptr , nullptr , nullptr , nullptr);
		loadTexture(scene , &mesh_material , specular , specular_texture , Texture::SPECULAR , texture_database); 
	}
	else
		dummy_textures_type.push_back(Texture::SPECULAR); 	
	
	if(material->GetTextureCount(aiTextureType_EMISSIVE) > 0){
		material->GetTexture(aiTextureType_EMISSIVE , 0 , &emissive_texture , nullptr , nullptr , nullptr , nullptr , nullptr);
		mesh_material.setEmissiveFactor(10.f); 
		loadTexture(scene , &mesh_material , emissive , emissive_texture , Texture::EMISSIVE , texture_database); 
	}
	else
		dummy_textures_type.push_back(Texture::EMISSIVE);
	
	for(auto it = dummy_textures_type.begin(); it != dummy_textures_type.end() ; it ++)
		loadTextureDummy(&mesh_material , *it , texture_database);
	
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

std::pair<unsigned , Material> loadMaterials(const aiScene* scene , const aiMaterial* material , TextureDatabase* texture_database , unsigned id){	
	Material mesh_material = loadAllTextures(scene , material , texture_database);	
	float transparency_factor = loadTransparencyValue(material);  	
	mesh_material.setTransparency(transparency_factor);
	return std::pair<unsigned , Material> (id , mesh_material) ; 
}

/**
 * The function loads shader files and adds them to a shader database.
 */
void Loader::loadShaderDatabase(){	
	ShaderDatabase* shader_database = resource_database->getShaderDatabase();
	std::string vertex_shader = loadShader("../shaders/phong.vert") ; 
	std::string fragment_shader = loadShader("../shaders/phong.frag");
	std::string vertex_shader_pbr = loadShader("../shaders/pbr.vert"); 
	std::string fragment_shader_pbr = loadShader("../shaders/pbr.frag");  	
	std::string vertex_shader_cubemap = loadShader("../shaders/cubemap.vert"); 
	std::string fragment_shader_cubemap = loadShader("../shaders/cubemap.frag");
	std::string vertex_shader_bouding_box = loadShader("../shaders/bbox.vert"); 
	std::string fragment_shader_bounding_box = loadShader("../shaders/bbox.frag"); 
	std::string vertex_shader_screen_fbo = loadShader("../shaders/screen_fbo.vert"); 
	std::string fragment_shader_screen_fbo = loadShader("../shaders/screen_fbo.frag"); 	
	std::string vertex_envmap_to_cubemap = loadShader("../shaders/envmap_bake.vert"); 
	std::string fragment_envmap_to_cubemap = loadShader("../shaders/envmap_bake.frag");
	std::string fragment_irradiance_compute = loadShader("../shaders/irradiance_baker.frag");
	std::string fragment_envmap_prefilter = loadShader("../shaders/envmap_prefilter.frag");
	std::string fragment_brdf_lut_baker = loadShader("../shaders/brdf_lookup_table_baker.frag"); 
	shader_database->addShader(vertex_shader_bouding_box , fragment_shader_bounding_box , Shader::BOUNDING_BOX);  
	shader_database->addShader(vertex_shader , fragment_shader , Shader::BLINN) ; 
	shader_database->addShader(vertex_shader_cubemap , fragment_shader_cubemap , Shader::CUBEMAP) ; 
	shader_database->addShader(vertex_shader_screen_fbo , fragment_shader_screen_fbo , Shader::SCREEN_FRAMEBUFFER); 
	shader_database->addShader(vertex_shader_pbr , fragment_shader_pbr , Shader::PBR); 
	shader_database->addShader(vertex_envmap_to_cubemap , fragment_envmap_to_cubemap , Shader::ENVMAP_CUBEMAP_CONVERTER); 
	shader_database->addShader(vertex_envmap_to_cubemap , fragment_irradiance_compute , Shader::IRRADIANCE_CUBEMAP_COMPUTE); 
	shader_database->addShader(vertex_envmap_to_cubemap , fragment_envmap_prefilter , Shader::ENVMAP_PREFILTER);
	shader_database->addShader(vertex_envmap_to_cubemap , fragment_brdf_lut_baker , Shader::BRDF_LUT_BAKER);
}



/**
 * The function `aiMatrix4x4ToGlm` converts an `aiMatrix4x4` object to a `glm::mat4` object in C++.
 * 
 * @param from The "from" parameter is of type aiMatrix4x4, which is a 4x4 matrix structure used in the
 * Assimp library. It represents a transformation matrix.
 * 
 * @return a glm::mat4 object.
 */
inline glm::mat4 aiMatrix4x4ToGlm(const aiMatrix4x4& from){ //https://stackoverflow.com/a/29184538
    glm::mat4 to;
    to[0][0] = (GLfloat)from.a1; to[0][1] = (GLfloat)from.b1;  to[0][2] = (GLfloat)from.c1; to[0][3] = (GLfloat)from.d1;
    to[1][0] = (GLfloat)from.a2; to[1][1] = (GLfloat)from.b2;  to[1][2] = (GLfloat)from.c2; to[1][3] = (GLfloat)from.d2;
    to[2][0] = (GLfloat)from.a3; to[2][1] = (GLfloat)from.b3;  to[2][2] = (GLfloat)from.c3; to[2][3] = (GLfloat)from.d3;
    to[3][0] = (GLfloat)from.a4; to[3][1] = (GLfloat)from.b4;  to[3][2] = (GLfloat)from.c4; to[3][3] = (GLfloat)from.d4;
    return to;
}

/**
 * The function "fillTreeData" recursively fills a tree data structure with scene nodes and their
 * corresponding transformations and meshes.
 * 
 * @param ai_node A pointer to an aiNode object, which represents a node in the scene hierarchy of an
 * imported model. This parameter is used to traverse the scene hierarchy and extract data from each
 * node.
 * @param mesh_lookup The `mesh_lookup` parameter is a vector that contains pointers to `Mesh` objects.
 * These `Mesh` objects represent the meshes in the scene. The `ai_node->mMeshes` array contains
 * indices that correspond to the meshes in the `mesh_lookup` vector.
 * @param parent The parent parameter is a pointer to the parent scene node. It represents the parent
 * node in the scene graph hierarchy.
 * @param node_deletion The `node_deletion` parameter is a vector that stores pointers to
 * SceneNodeInterface objects that need to be deleted later by the SceneTree structure. 
 * 
 * @return a pointer to a random node of the tree... Root can be determined from any point of the tree , by using the method SceneNodeInterface::returnRoot()
 */
SceneNodeInterface* fillTreeData(aiNode *ai_node , const std::vector<Mesh*>& mesh_lookup , SceneNodeInterface* parent , std::vector<SceneNodeInterface*> &node_deletion){
	if(ai_node != nullptr){
		std::string name = ai_node->mName.C_Str();
		glm::mat4 transformation = aiMatrix4x4ToGlm(ai_node->mTransformation);
		std::vector<SceneNodeInterface*> add_node ; 
		if(ai_node->mNumMeshes == 0){
			add_node.push_back(SceneNodeBuilder::buildEmptyNode(parent));
			node_deletion.push_back(add_node[0]); 
		}	 
		else if(ai_node->mNumMeshes == 1)
			add_node.push_back(mesh_lookup[ai_node->mMeshes[0]]);
		else{
			add_node.push_back(SceneNodeBuilder::buildEmptyNode(parent)); //Little compatibility hack between assimp and the node system, assimp nodes can contain multiples meshes , but SceneTreeNode can be a mesh. 
			node_deletion.push_back(add_node[0]); 						 //So we create a dummy node at position 0 in add_node to be the ancestors of the children nodes , while meshes will be attached to parent and without children.
			for(unsigned i = 0 ; i < ai_node->mNumMeshes ; i++)
				add_node.push_back(mesh_lookup[ai_node->mMeshes[i]]); 	
		}
		for(auto A : add_node){
			A->setLocalModelMatrix(transformation);
			A->setName(name);
			std::vector<SceneNodeInterface*> parents_array = {parent}; 
			A->setParents(parents_array); 	
		}
		for(unsigned i = 0 ; i < ai_node->mNumChildren ; i++)
			fillTreeData(ai_node->mChildren[i] , mesh_lookup , add_node[0] , node_deletion); 
		return add_node[0]; 
	}
	return nullptr ; 
}

/**
 * The function generates a scene tree from a model scene and a lookup of mesh nodes.
 * 
 * @param modelScene The modelScene parameter is a pointer to an aiScene object, which represents a 3D
 * model scene loaded from a file using the Assimp library. It contains information about the model's
 * hierarchy, meshes, materials, textures, animations, etc.
 * @param node_lookup The `node_lookup` parameter is a vector of pointers to `Mesh` objects. It is used
 * to map the `aiNode` objects in the `modelScene` to their corresponding `Mesh` objects.
 * 
 * @return a SceneTree object.
 */
SceneTree generateSceneTree(const aiScene* modelScene , const std::vector<Mesh*> &node_lookup){	
	aiNode *ai_root = modelScene->mRootNode ;
	SceneTree scene_tree;
	std::vector<SceneNodeInterface*> node_deletion ;
	SceneNodeInterface* node = fillTreeData(ai_root , node_lookup , nullptr , node_deletion);
	assert (node != nullptr) ;  
	node = node->returnRoot(); 
	scene_tree.setRoot(node); 
	scene_tree.updateAccumulatedTransformations(); 
	return scene_tree; 
}

/**
 * The function "load_geometry_buffer" takes an array of aiVector3D objects, converts them to a
 * specified dimension, and stores the result in a destination vector.
 * 
 * @param dest The `dest` parameter is a reference to a `std::vector<T>` object, where `T` is the type
 * of the elements in the vector. This vector will be used to store the loaded geometry data.
 * @param from The "from" parameter is a pointer to an array of aiVector3D objects.
 * @param size The size parameter represents the number of elements in the from array.
 * @param dimension The "dimension" parameter determines the number of components in each vector. If
 * dimension is set to 3, each vector will have three components (x, y, and z). If dimension is set to
 * 2, each vector will have two components (x and y).
 */
template<class T>
void load_geometry_buffer(std::vector<T> &dest , const aiVector3D *from , int size , int dimension ){
	for(int f = 0 , i = 0 ; f < size ; f++){	
		const aiVector3D vect = from[f]; 
		if(dimension == 3){
			dest[i] = vect.x; 
			dest[i + 1] = vect.y; 
			dest[i + 2] = vect.z;
			i += 3 ; 
		}
		else {
			dest[i] = vect.x;
			dest[i + 1] = vect.y;
			i += 2;  
		}	
	} 
}

/**
 * The function `load_indices_buffer` loads the indices of a mesh's faces into a destination vector.
 * 
 * @param dest A reference to a vector of unsigned integers where the indices will be loaded into.
 * @param faces The `faces` parameter is a pointer to an array of `aiFace` objects. Each `aiFace`
 * object represents a face in a 3D model and contains an array of indices that define the vertices of
 * the face.
 * @param num_faces The parameter "num_faces" represents the number of faces in the mesh.
 */
void load_indices_buffer(std::vector<unsigned> &dest , const aiFace *faces , int num_faces){
	for(int i = 0 , f = 0 ; i < num_faces ; i++ , f += 3){
		assert(faces[i].mNumIndices==3) ;  
		dest[f] = faces[i].mIndices[0]; 	
		dest[f + 1] = faces[i].mIndices[1]; 	
		dest[f + 2] = faces[i].mIndices[2]; 	
	}
}


/**
 * The function "geometry_fill_buffers" fills the buffers of a 3D object with vertex, normal, tangent,
 * bitangent, and UV data from an imported model.
 * 
 * @param modelScene modelScene is a pointer to an aiScene object, which represents a 3D model scene.
 * It contains information about the model's meshes, materials, textures, and other properties.
 * @param i The parameter "i" is the index of the mesh in the model scene that you want to fill the
 * buffers for.
 * 
 * @return a std::pair<unsigned, Object3D*>.
 */
std::pair<unsigned , Object3D*> geometry_fill_buffers(const aiScene* modelScene , unsigned i){			
	const aiMesh* mesh = modelScene->mMeshes[i] ; 
	Object3D *object = new Object3D ;
	auto size_dim3 = mesh->mNumVertices * 3; 
	auto size_dim2 = mesh->mNumVertices * 2; 
	auto size_dim3_indices = mesh->mNumFaces * 3; 
	object->vertices.resize(size_dim3);
	object->normals.resize(size_dim3); 
	object->tangents.resize(size_dim3); 
	object->bitangents.resize(size_dim3);	
	object->indices.resize(size_dim3_indices); 
	object->uv.resize(size_dim2) ; 	
	assert(mesh->HasTextureCoords(0)) ; 
	std::future<void> f_vertices , f_normals , f_bitangents , f_tangents , f_uv ;
	f_vertices = std::async(std::launch::async, [&](){
			load_geometry_buffer(object->vertices , mesh->mVertices , mesh->mNumVertices , 3);
			}); 
	f_normals = std::async(std::launch::async,[&](){
			load_geometry_buffer(object->normals , mesh->mNormals , mesh->mNumVertices , 3);
			}); 
	f_bitangents = std::async(std::launch::async,[&](){
			load_geometry_buffer(object->bitangents , mesh->mBitangents , mesh->mNumVertices , 3);
			});
	f_tangents = std::async(std::launch::async,[&](){
			load_geometry_buffer(object->tangents , mesh->mTangents , mesh->mNumVertices , 3);
			}); 
	f_uv = std::async(std::launch::async,[&](){
			load_geometry_buffer(object->uv , mesh->mTextureCoords[0] , mesh->mNumVertices , 2);
			});	
	load_indices_buffer(object->indices , mesh->mFaces , mesh->mNumFaces);	
	std::vector<std::shared_future<void>> shared_futures = {f_vertices.share() , f_normals.share() , f_bitangents.share() , f_tangents.share() , f_uv.share()};			
	std::for_each(shared_futures.begin() , shared_futures.end() , [](std::shared_future<void> &it) -> void{	
		it.wait(); 
	});
	return std::pair<unsigned , Object3D*>(i , object); 	
};




/**
 * The function "loadObjects" loads objects from a file using the Assimp library and returns a pair
 * containing a vector of meshes and a scene tree.
 * 
 * @param file The "file" parameter is a const char pointer that represents the file path of the scene
 * file that needs to be loaded.
 * 
 * @return a pair containing a vector of Mesh pointers and a SceneTree object.
 */

std::pair<std::vector<Mesh*> , SceneTree> Loader::loadObjects(const char* file){ //TODO! return scene data structure , with lights + meshes + cameras 
	TextureDatabase *texture_database = resource_database->getTextureDatabase() ; 	
	ShaderDatabase *shader_database = resource_database->getShaderDatabase(); 
	std::pair<std::vector<Mesh*> , SceneTree> objects;
	Assimp::Importer importer ;	
	std::mutex mesh_load_mutex;	
	const aiScene *modelScene = importer.ReadFile(file , aiProcess_CalcTangentSpace | aiProcess_Triangulate  | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs ) ;
	if(modelScene != nullptr){	
		std::vector<std::future<std::pair<unsigned ,Object3D*>>> loaded_meshes_futures ; 	
		std::vector<Material> material_array ;	
		std::vector<Mesh*> node_lookup_table; 	
		Shader* shader_program = shader_database->get(Shader::PBR) ; 	
		node_lookup_table.resize(modelScene->mNumMeshes); 
		material_array.resize(modelScene->mNumMeshes);
		for(unsigned int i = 0 ; i < modelScene->mNumMeshes ; i++){ 
			loaded_meshes_futures.push_back(
				std::async(std::launch::async , geometry_fill_buffers , modelScene,  i) //*We launch multiple threads loading the geometry , and the main thread loads the materials
			);
			aiMaterial* ai_mat = modelScene->mMaterials[modelScene->mMeshes[i]->mMaterialIndex]; 	
			material_array[i] = loadMaterials(modelScene , ai_mat , texture_database , i).second; 
		}	
		for(auto it = loaded_meshes_futures.begin(); it != loaded_meshes_futures.end() ; it++ ){		
			std::pair<unsigned , Object3D*> geometry_loaded = it->get();	
			unsigned mesh_index = geometry_loaded.first ;
			Object3D* geometry = geometry_loaded.second ;
			const aiMesh* mesh = modelScene->mMeshes[mesh_index] ; 
			const char* mesh_name = mesh->mName.C_Str(); 
			std::string name(mesh_name);	
			Mesh *loaded_mesh = static_cast<Mesh*>(SceneNodeBuilder::buildMesh(nullptr , name , std::move(*geometry), material_array[mesh_index] , shader_program )) ;
			std::cout << "object loaded : " << name << "\n" ;
			node_lookup_table[mesh_index] = loaded_mesh; 	
			objects.first.push_back(loaded_mesh);
			delete geometry ; 	
		}
		SceneTree scene_tree = generateSceneTree(modelScene , node_lookup_table); 
		objects.second = scene_tree ; 	
		return objects ; 
	}	
	else{
		std::cout << "Problem loading scene" << std::endl ; 
		return std::pair<std::vector<Mesh*> , SceneTree>() ; 
	}
	
}

/**
 * The function loads a file and generates a vector of meshes, including a cube map if applicable.
 * 
 * @param file A pointer to a character array representing the file path of the 3D model to be loaded.
 * 
 * @return A vector of Mesh pointers.
 */
std::pair<std::vector<Mesh*> , SceneTree> Loader::load(const char* file){
	TextureDatabase *texture_database = resource_database->getTextureDatabase(); 	
	ShaderDatabase *shader_database = resource_database->getShaderDatabase(); 
	texture_database->clean();
	shader_database->clean(); 
	loadShaderDatabase(); 
	std::pair<std::vector<Mesh*> , SceneTree> scene = loadObjects(file); 
	/*Mesh* cube_map = generateCubeMap(false) ; 		
	if(cube_map != nullptr){
		scene.first.push_back(cube_map);
		scene.second.setAsRootChild(cube_map); 
	}*/
	errorCheck(__FILE__ , __LINE__); 	
	return scene ; 
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


//TODO: [AX-21] Provide a way to choose the skybox texture on the UI
/**
 * This function generates a cube map mesh with textures loaded from the skybox folder.
 * 
 * @param num_textures The number of textures currently loaded in the texture database before adding
 * the cubemap texture.
 * @param is_glb The parameter "is_glb" is not used in the function and therefore has no effect on the
 * generated cube map.
 * 
 * @return A pointer to a Mesh object representing a cube map.
 */
Mesh* Loader::generateCubeMap(bool is_glb){ 
	Mesh *cube_map = new CubeMapMesh(); 
	Material material ; 
	ShaderDatabase *shader_database = resource_database->getShaderDatabase(); 
	TextureDatabase* texture_database = resource_database->getTextureDatabase(); 	
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
	cubemap.data_format = Texture::RGBA ; 
	cubemap.internal_format = Texture::RGBA ; 
	cubemap.data_type = Texture::UBYTE;  
	cubemap.width = left.width() ; 
	cubemap.height = left.height(); 	
	cubemap.data = new uint32_t [cubemap.width * cubemap.height * 6] ;	
	unsigned int k = 0 ; 
	uint32_t *pointer_on_cubemap_data = cubemap.data ; 	
	std::vector<std::future<void>> threads_future ;
	auto thread_lambda_func = [&array](unsigned i , uint32_t* pointer_on_cubemap_data, unsigned width , unsigned height) -> void {
		unsigned k = i * width * height ;  
		for(unsigned y = 0 ; y < height ; y++){		
			const QRgb *line = reinterpret_cast<const QRgb*>(array[i].scanLine(y)) ; //Since the cubemaps don't have alpha channel , we interpret the data as RGB  
			for(unsigned x = 0 ; x < width ; x++){
				uint32_t rgba = line[x] ; 
				int a = rgba >> 24 ;
				int r = (rgba >> 16) & 0xFF ; 
				int g = (rgba >> 8) & 0xFF ; 
				int b =  rgba ;	
				uint32_t rgba_final = (a << 24) | (b << 16) | (g << 8) | r ; 
				pointer_on_cubemap_data[k] = rgba_final ; 
				k++ ;		
			}
		}
	};
	for(unsigned i = 0 ; i < array.size() ; i ++)	
		threads_future.push_back(
			std::async(thread_lambda_func , i , pointer_on_cubemap_data , cubemap.width , cubemap.height) 
		);
	std::for_each(threads_future.begin() , threads_future.end() , [](std::future<void> &it){
		it.get(); 
	});
	unsigned index = texture_database->addTexture(&cubemap , Texture::CUBEMAP , false) ; 
	material.addTexture(index , Texture::CUBEMAP);
	cube_map->material = material ; 
	cube_map->setShader(shader_database->get(Shader::CUBEMAP)) ; 
	cubemap.clean(); 	
	return cube_map; 		
}



EnvironmentMap2DTexture* Loader::loadHdrEnvmap(){
	TextureDatabase* texture_database = resource_database->getTextureDatabase(); 
	std::string folder_night = "../Ressources/Skybox_Textures/HDR/Night_City/" ; 
	std::string folder_forest = "../Ressources/Skybox_Textures/HDR/Forest/" ;
	std::string folder_park = "../Ressources/Skybox_Textures/HDR/Park/" ;
	std::string folder_snow = "../Ressources/Skybox_Textures/HDR/Snow/" ;
	std::string folder_sky = "../Ressources/Skybox_Textures/HDR/Sky/" ; 
	std::string folder_street = "../Ressources/Skybox_Textures/HDR/Street/" ;  
	std::string env = folder_night + "night_env.hdr";
	std::string hdr = folder_night +"night.hdr";

	hdr = folder_forest + "Forest.hdr" ; 
	TextureData envmap; 	
	int width , height , channels ; 
	float *hdr_data = stbi_loadf(hdr.c_str() , &width , &height , &channels , 0);
	
	if(stbi_failure_reason())
		std::cout << stbi_failure_reason() << "\n"; 
	envmap.width = static_cast<unsigned int>(width); 
	envmap.height = static_cast<unsigned int>(height);
	envmap.data_type = Texture::FLOAT; 
	envmap.internal_format = Texture::RGB32F ; 
	envmap.data_format = Texture::RGB;
	envmap.nb_components = channels ;
	envmap.f_data = new float[width * height * channels]; 
	std::memcpy(envmap.f_data , hdr_data , width * height * channels * sizeof(float)); 	
	/* Furnace test */
	/*for(unsigned i = 0 ; i < width * height * channels ; i++)
		envmap.f_data[i] = 1.f ; */
	int index = texture_database->addTexture(&envmap , Texture::ENVMAP2D); 
	EnvironmentMap2DTexture* envmap_texture = dynamic_cast<EnvironmentMap2DTexture*>(texture_database->get(index));
	envmap.clean(); 
	if(envmap_texture) 
		return envmap_texture; 	
	else{
		std::cout << "ENVMAP loading failed!\n" ; 
		return nullptr; 
	}

}
















}


