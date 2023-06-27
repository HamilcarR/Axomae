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
	for (auto A : shader_database){
		A.second->clean();
		delete A.second;
		A.second = nullptr;
	}
	shader_database.clear();
}

Shader *ShaderDatabase::addShader(const std::string vertex_code, const std::string fragment_code, const Shader::TYPE type){
	Mutex mutex ; 
	Mutex::Lock lock(mutex); 
	if (shader_database.find(type) == shader_database.end())
		shader_database[type] = ShaderFactory::constructShader(vertex_code, fragment_code, type);
	return shader_database[type];
}

void ShaderDatabase::initializeShaders(){
	for (auto A : shader_database)
		A.second->initializeShader();
}

bool ShaderDatabase::contains(const int t) {
	Mutex mutex; 
	Mutex::Lock lock(mutex); 
	const Shader::TYPE type = static_cast<Shader::TYPE>(t);
	return shader_database.find(type) != shader_database.end();
}

Shader *ShaderDatabase::get(const int t) {
	Mutex mutex; 
	Mutex::Lock lock(mutex); 	
	const Shader::TYPE type = static_cast<Shader::TYPE>(t);
	auto it = shader_database.find(type);
	if (it != shader_database.end())
		return it->second;
	else
		return nullptr;
}

void ShaderDatabase::recompile(){
	for (auto A : shader_database)
		A.second->recompile();	
}
