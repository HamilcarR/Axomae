#ifndef SCENEDATA_H
#define SCENEDATA_H

#include "constants.h"
#include "utils_3D.h"
#include "Loader.h"
#include "Mesh.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_3_Core> 
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject> 
#include <QDebug>
#include <QString> 
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture> 




class SceneData {
	public:
		SceneData();
		virtual ~SceneData();	
		void initialize();
		bool prep_draw();
		void end_draw();
		void setNewScene(std::vector<axomae::Mesh> &new_scene);

		QOpenGLShaderProgram *shader_program ; 	
		QOpenGLVertexArrayObject vao ; 
		QOpenGLBuffer vertex_buffer ; 
		QOpenGLBuffer normal_buffer ; 
		QOpenGLBuffer index_buffer ; 
		QOpenGLBuffer texture_buffer ;
		QOpenGLBuffer color_buffer ;
		unsigned int sampler2D ; 
		axomae::Mesh *current_scene;
		bool start_draw ; 
		
};

inline void errorCheck(){
	GLenum error = GL_NO_ERROR;
	do {
    		error = glGetError();
    		if (error != GL_NO_ERROR) 
        		std::cout << "Error:" << error << std::endl ; 
	} while (error != GL_NO_ERROR);

}

inline void shaderErrorCheck(QOpenGLShaderProgram* shader_program){
	if(shader_program){
		QString log = shader_program->log(); 
		std::cout << log.constData() << std::endl; 
	}
}



#endif
