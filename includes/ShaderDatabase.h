#ifndef SHADERDATABASE_H
#define SHADERDATABASE_H

#include <map> 

#include "utils_3D.h"
#include "Shader.h"

class ShaderDatabase{
public:
	static ShaderDatabase* getInstance(); 
	void clean(); 
	void destroy(); 
	static bool isInstanced(){return instance != nullptr;}		
	
	/* compare the fragment / vertex / geom / tesselation code of shader to each instance in shader_database ... 
	 * if an identical shader is found , return it's address . If not , store the new shader in the database , and returns it's address 
	 */ 
	Shader* addShader(const std::string vertex_code , const std::string fragment_code ,const Shader::TYPE type);  
	bool contains(const Shader::TYPE type); 
	Shader* get(Shader::TYPE type) const ; 
private:
	ShaderDatabase(); 
	virtual ~ShaderDatabase() ;
	

private:
	static ShaderDatabase* instance; 
	std::map<int, Shader*> shader_database ; 


}; 


#endif 
