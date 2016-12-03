#ifndef TERMOPT_H
#define TERMOPT_H

#include "ImageManager.h"
#include "ImageImporter.h"
#include "Window.h"
#include "Renderer.h" 
#include <memory> 


namespace maptomix{


	
	struct Validation {
		bool validated ; 
		std::vector <std::string> command_arguments ; 


	};





	static const char *prompt[] ={
	">"," :"


	};

	static const char *command[] = {
		"window",	//create a new window
		"normalMap",	//compute normal map
		"heightMap",	//compute height map
		"dudv" , 	//compute DUDV
		"save" ,	//save image
		"contrast",	//set contrast
		"exit",		//exit the program
		"render",	//render an object on a OpenGL window
		"load"		//load an image
		};
	
	
	

	enum: unsigned  {LOAD = 8 , RENDER = 7 , EXIT = 6 , CONTRAST = 5 , SAVE = 4 , DUDV = 3 , HMAP = 2 , NMAP = 1 , WIN = 0 };
	enum: unsigned	{WIN_ARGS = 3 };
	
	enum: unsigned  {RED = 0 , BLUE = 1 , GREEN = 2 , YELLOW = 3 , RESET = 4 };


	static const char *colors[]={
		"\033[31m",
		"\033[34m",
		"\033[32m",
		"\033[33m",
		"\033[0m"


	};


/*******************************************************************************************************************************************************/
	void process_command(std::string user_input); 

	



	class ProgramStatus{
		public:
			
			static ProgramStatus* getInstance(){ if(instance==nullptr) instance = new ProgramStatus();return instance;}
			static void	       Quit(){if(instance != nullptr) delete instance;instance=nullptr;}
			
		private:
			/*functions*/
			ProgramStatus();
			~ProgramStatus(); 
			


			/*attributes*/
			std::vector<std::shared_ptr<Window>> display; 
			std::vector<std::shared_ptr<Renderer>> renderer ; 
			static ProgramStatus* instance; 
				
		



	};

}




#endif
