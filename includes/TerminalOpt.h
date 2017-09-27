#ifndef TERMOPT_H
#define TERMOPT_H
#include "EventHandler.h"
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
	">"," :" , "=:" , ">>" 


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
	
	
	

	enum: unsigned  {CHK_CURRENT_IMG = 11 , SELECT = 10 , LISTIDS = 9 , LOAD = 8 , RENDER = 7 , EXIT = 6 , CONTRAST = 5 , SAVE = 4 , DUDV = 3 , HMAP = 2 , NMAP = 1 , WIN = 0 };
	enum: unsigned	{WIN_ARGS = 3 };
	

#ifdef __unix__
	enum : unsigned { RED = 0, BLUE = 1, GREEN = 2, YELLOW = 3, RESET = 4 };

	static const char *colors[] = {
		"\033[31m",
		"\033[34m",
		"\033[32m",
		"\033[33m",
		"\033[0m"


};

#elif defined (WIN32) || defined(_WIN32)
	enum : unsigned { RED = 4, BLUE = 1, GREEN = 2, YELLOW = 6, RESET = 10 };

/*Color Codes:
0 = Black
1 = Blue
2 = Green
3 = Aqua
4 = Red
5 = Purple
6 = Yellow
7 = White
8 = Gray
9 = Light Blue
A = Light Green
B = Light Aqua
C = Light Red
D = Light Purple
E = Light Yellow
F = Bright White
*/


#endif
	

/*******************************************************************************************************************************************************/


	



	class ProgramStatus{
		public:
			
			static ProgramStatus* getInstance(){ if(instance==nullptr) instance = new ProgramStatus();return instance;}
			static void	       Quit(){if(instance != nullptr) delete instance;instance=nullptr;}
			
			void process_command(std::string user_input); 
			static void loop_thread(void* window);
			void setEvent(SDL_Event &ev){event = ev;}
		private:
			/*functions*/
			ProgramStatus();
			~ProgramStatus(); 
			


			/*attributes*/
			std::vector < std::pair< SDL_Surface* , std::string>> images;
			std::unique_ptr<Window> display; 
			std::shared_ptr<Renderer> renderer ; 
			int _idCurrentImage; 
			static ProgramStatus* instance; 

			SDL_Event event; 
						
		



	};

}




#endif
