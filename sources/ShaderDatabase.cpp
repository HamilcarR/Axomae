#include "../includes/ShaderDatabase.h"
#include "../includes/ShaderFactory.h"
#include "../includes/Mutex.h"

ShaderDatabase::ShaderDatabase(){
}

ShaderDatabase::~ShaderDatabase(){
}

void ShaderDatabase::purge(){
	clean();
}

void ShaderDatabase::clean(){
	Mutex::Lock lock(mutex);
	for (auto A : shader_database){
		A.second->clean();
		delete A.second;
		A.second = nullptr;
	}
	shader_database.clear();
}

Shader *ShaderDatabase::addShader(const std::string vertex_code, const std::string fragment_code, const Shader::TYPE type){
	Mutex::Lock lock(mutex); 
	if (shader_database.find(type) == shader_database.end())
		shader_database[type] = ShaderFactory::constructShader(vertex_code, fragment_code, type);
	return shader_database[type];
}

void ShaderDatabase::initializeShaders(){
	Mutex::Lock lock(mutex);
	for (auto A : shader_database)
		A.second->initializeShader();
}

bool ShaderDatabase::contains(const Shader::TYPE type) {
	Mutex::Lock lock(mutex); 
	return shader_database.find(type) != shader_database.end();
}

Shader *ShaderDatabase::get(const Shader::TYPE type) {
	Mutex::Lock lock(mutex); 	
	auto it = shader_database.find(type);
	if (it != shader_database.end())
		return it->second;
	else
		return nullptr;
}

void ShaderDatabase::recompile(){
	Mutex::Lock lock(mutex);
	for (auto A : shader_database)
		A.second->recompile();	
}

Shader::TYPE ShaderDatabase::add(Shader* shader , bool keep){
	Mutex::Lock lock(mutex); 
	for(auto A : shader_database)
		if(A.second == shader)
			return A.first; 
	shader_database[shader->getType()] = shader ;
	return shader->getType(); 
}

std::pair<Shader::TYPE , Shader*> ShaderDatabase::contains(const Shader* shader){
	Mutex::Lock lock(mutex); 
	for(auto A : shader_database){
		if(A.second == shader)
			return A; 
	}
	return std::pair(Shader::EMPTY , nullptr); 
}