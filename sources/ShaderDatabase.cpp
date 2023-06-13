#include "../includes/ShaderDatabase.h" 
#include "../includes/ShaderFactory.h"

ShaderDatabase* ShaderDatabase::instance = nullptr; 



ShaderDatabase* ShaderDatabase::getInstance(){
	if( instance == nullptr)
		instance = new ShaderDatabase() ; 
	return instance ; 
}

void ShaderDatabase::destroy(){
	if(instance != nullptr)
		delete instance ; 
	instance = nullptr; 
}

void ShaderDatabase::clean(){
	for(auto A : shader_database){
		A.second->clean(); 
		delete A.second ; 
		A.second = nullptr ; 
	} 
	shader_database.clear() ; 
}

Shader* ShaderDatabase::addShader(const std::string vertex_code ,const std::string fragment_code ,const Shader::TYPE type){
	if(!contains(type))
		shader_database[type] = ShaderFactory::constructShader(vertex_code , fragment_code , type) ; 
	return shader_database[type] ; 
}

void ShaderDatabase::initializeShaders(){
	for(auto A : shader_database)
		A.second->initializeShader();
}

bool ShaderDatabase::contains(const Shader::TYPE type){
	return shader_database.find(type) != shader_database.end(); 
}


Shader* ShaderDatabase::get(const Shader::TYPE type) const {
	auto it = shader_database.find(type);
	if(it != shader_database.end())
		return it->second ;
	else
		return nullptr;
}

void ShaderDatabase::recompile(){
	for(auto A : shader_database){
		A.second->recompile(); 
	}
}

ShaderDatabase::ShaderDatabase(){

}

ShaderDatabase::~ShaderDatabase(){

}
