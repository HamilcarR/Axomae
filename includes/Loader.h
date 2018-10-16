#ifndef LOADER_H
#define LOADER_H

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "constants.h"
#include <cstdlib>
#include <assert.h>
#include <iostream>
#include <memory>

/*3D obj loader*/

namespace axomae{

	class Loader {

		public:

			static Loader* getInstance(); 	
			static std::vector<Object3D> load(const char* file);   				
			static void close(); 	
		
		private: 
			Loader();	
			~Loader();
			static Loader* instance; 
	
	
	
	};








}









#endif
