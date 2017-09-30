#include "../includes/ImageImporter.h"
#include "../includes/TerminalOpt.h"
#include <regex>
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <future>
#include <cctype>
namespace maptomix{
	
	const int STRING_SIZE = 256 ; 

	ProgramStatus *ProgramStatus::instance = nullptr; 
	
	static void print(std::string&, int8_t); 
	static const std::regex command_regex[]={
		std::regex("window [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),	//open a new SDL window
		std::regex("nmap [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),	//generate normal map (from height map)
		std::regex("hmap [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),	//generate height map (from albedo texture) 
		std::regex("dudv [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),	//generate dudv map
		std::regex("save (/?([a-zA-Z0-9]+)/?)*[a-zA-Z0-9]+.[a-zA-Z0-9]+ (/?([a-zA-Z0-9]+)/?)*[a-zA-Z0-9]+.[a-zA-Z0-9]+",std::regex_constants::icase),	// save image on the disk
		std::regex("contrast [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase), // set contrast
		std::regex("exit [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase), //exit the app
		std::regex("render [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase), //render the texture on a mesh
		std::regex("load (/?([a-zA-Z0-9]+)/?)*[a-zA-Z0-9]+.[a-zA-Z0-9]+",std::regex_constants::icase),	//load an image
		std::regex("ls" , std::regex_constants::icase),		//list all image ids
		std::regex("select [0-9]+",std::regex_constants::icase), //select image id as work image
		std::regex("crtID" , std::regex::icase)			//check current image id 


	};

/*******************************************************************************************************************************************************/
	static bool check_if_number(std::string& input) {
		
		bool it = std::all_of(input.begin(),input.end(), [](char c){ return std::isdigit(c);});
		return it;

	}

/*******************************************************************************************************************************************************/

	/*retrieve an argument from a command*/
	static std::string get_word(std::string& input , Uint8 pos){


		if(pos > input.size() || input.size() == 0 ) 
		      return std::string();
		else{

			char input_c[STRING_SIZE] = " "; 
		        strcpy(input_c,input.c_str()) ; 	
			char * tokens = strtok(input_c , " \n" );
			int number_args = 0 ; 
		       	while(tokens){
				if(pos == number_args)
					return std::string(tokens); 
			
					tokens = strtok(NULL , " \n" ); 
					number_args ++; 	
				

			}
			return std::string() ; 


		}	


	}



/*******************************************************************************************************************************************************/

	static Validation validate_command_load(std::string input){
		std::string delimiter = " " ;

		if(std::regex_match(input , command_regex[LOAD])){

				std::vector<std::string> arg_array ; 
				std::string arg1 = get_word(input,1);

				if(arg1.size()>0)
				{
					arg_array.push_back(arg1);
					return {true , arg_array}; 

				}
				else
					return {false,std::vector<std::string>()}; 



		}			

		else
		return {false,std::vector<std::string>()}; 
	}

/*******************************************************************************************************************************************************/
	static Validation validate_command_save(std::string input){


		return {false,std::vector<std::string>()}; 
	}


/*******************************************************************************************************************************************************/
	static Validation validate_command_window(std::string input){
		std::string delimiter = " " ; 
		if(std::regex_match(input,command_regex[WIN])){
			bool validate_input = false;
		        std::vector<std::string> arg_array;	
			std::string arg1=get_word(input,1);
			std::string arg2=get_word(input,2);
			std::string arg3=get_word(input,3);
		     	bool c1 = check_if_number(arg1);
			bool c2 = check_if_number(arg2);
			if(c1 && c2 && arg1.size()>0 && arg2.size()>0 && arg3.size()>0){
				arg_array.push_back(arg1); 
				arg_array.push_back(arg2); 
				arg_array.push_back(arg3); 
				 return {true,arg_array};


			}		
			else
				return {false,std::vector<std::string>()}; 
			
						
		}
		else
			return {false,std::vector<std::string>()}; 

	}

/*******************************************************************************************************************************************************/
	static Validation validate_command_contrast(std::string input){

		return {false,std::vector<std::string>()}; 

	}

/*******************************************************************************************************************************************************/

	static Validation validate_command_dudv(std::string input){
		std::string delimiter = " " ; 
			

		return {false,std::vector<std::string>()}; 

	}

/*******************************************************************************************************************************************************/
	static Validation validate_command_nmap(std::string input){

		return {false,std::vector<std::string>()}; 

	}



/*******************************************************************************************************************************************************/

	static Validation validate_command_hmap(std::string input){
		std::string delimiter = " " ; 
			


		return {false,std::vector<std::string>()}; 
	}

/*******************************************************************************************************************************************************/
	static Validation validate_command_render(std::string input){

		return {false,std::vector<std::string>()}; 
	}

/*******************************************************************************************************************************************************/

	static Validation validate_command_select(std::string input) {
		std::string w1 = get_word(input, 1);

		if (std::regex_match(input, command_regex[SELECT])) {
			if (check_if_number(w1)) {
				std::vector<std::string> A;
				A.push_back(w1);
				return { true , A };
			}
			else
				return { false , std::vector<std::string>() };

		}

		else
			return { false , std::vector<std::string>() };


	}


/*******************************************************************************************************************************************************/
	static void loop_thread(int window , int threadID) {
		ProgramStatus *instance = ProgramStatus::getInstance(); 
		std::vector < std::pair< SDL_Surface*, std::string>> images = instance->getImages(); 
		std::vector<std::pair<std::shared_ptr<Window>, SDL_Event>> windows = instance->getWindows(); 

		int _idCurrentImage = instance->getCurrentImageId(); 
			
		if (window >= 0 && window < windows.size) {
			SDL_Event event = windows[window].second;

		}
		else
			return; 
	}

/*******************************************************************************************************************************************************/
	void ProgramStatus::create_window(int width , int height , const char* name) {
		std::shared_ptr<Window> display = std::shared_ptr<Window>(new Window(width, height, name));
		SDL_Event event; 
		display->setEvent(event);
		windows.push_back(std::pair<std::shared_ptr<Window>, SDL_Event>(display, event));
		
	}
	

/*******************************************************************************************************************************************************/
	void ProgramStatus::process_command(std::string user_input){


		/*does it match ?*/

		bool save = std::regex_search(user_input,command_regex[SAVE]);
		bool load= std::regex_search(user_input,command_regex[LOAD]);
		bool window = std::regex_search(user_input,command_regex[WIN]);
		bool normalmap = std::regex_search(user_input,command_regex[NMAP]);
		bool heightmap = std::regex_search(user_input,command_regex[HMAP]);
		bool contrast = std::regex_search(user_input,command_regex[CONTRAST]);
		bool render = std::regex_search(user_input,command_regex[RENDER]);
		bool dudv = std::regex_search(user_input,command_regex[DUDV]);
		bool closew = std::regex_search(user_input,std::regex("exwin"));
		bool list_ids = std::regex_search(user_input, command_regex[LISTIDS]); 
		bool selectid = std::regex_search(user_input, command_regex[SELECT]); 

		if(save){
			/*create a "state" class with pointers to every elements: Save the image of the window if a window exists*/

		}
		else if(load){
			Validation v = validate_command_load(user_input);
			if (v.validated)
			{
				std::cout << "File : " << v.command_arguments[0] << " loading..." << "\n" ; 
				ImageImporter *instance = ImageImporter::getInstance(); 
				SDL_Surface* im = instance->load_image(static_cast<const char*>(v.command_arguments[0].c_str()));
				if(im)
					images.push_back(std::pair<SDL_Surface* , std::string>(im , v.command_arguments[0]));				
				


			}
			else
			{
				std::cout <<"Wrong command used !" << "\n" ; 

			}
		
		
		}
		else if(window){
			Validation v = validate_command_window(user_input); 
			if(v.validated)
			{
				int w = std::stoi(v.command_arguments[0]) , h = std::stoi(v.command_arguments[1]) ; 
				std::string window_name = v.command_arguments[2];
				//void (ProgramStatus::*ref)(int , int , const char*) = &ProgramStatus::loop_thread;
				//auto thread = std::async(std::launch::async, ref, this, w, h, window_name.c_str());
				// std::thread(ref, this, w, h, window_name.c_str()).detach();
				create_window(w, h, window_name.c_str()); 
				

			}
			else
				std::cout << "wrong command"<<std::endl;



		}
		else if(normalmap){
		
		}
		else if(heightmap){
		
		}
		else if(contrast){
		}
		else if(render){
		
		}
		else if(dudv){
		
		}
		else if (list_ids) {
			if (images.size() == 0)
				print(std::string("No image loaded..."), RED);
			else {
				std::string liste = " ";
				unsigned int count = 0;

				for (std::pair<SDL_Surface*, std::string> p : images) {
					liste += std::to_string(count) + " : " + p.second + "\n";
					count++;
				}
				print(liste, BLUE); 
		

			}
		
		}
		
		else if(closew ){
		
			std::puts("Exiting...\n"); 
		//	display.reset(); 

		}
		else if (selectid) {

			Validation v = validate_command_select(user_input); 
			if (v.validated) {
				try {
					unsigned int id = std::stoi(v.command_arguments[0].c_str());
					if (id >= images.size() || id < 0) {
						print(std::string("invalid id input"), RED); 
					}
					else
						_idCurrentImage = id;

				}
				catch (std::invalid_argument  e) {
					print(std::string("invalid argument input"), RED); 

				}
			}
			

		}
		else {
			print(std::string("Wrong command ! "), RED); 

		}

	
	}




	static void print(std::string &arg, int8_t color) {
#ifdef __unix__
		std::cout << colors[color] << "\n"; 
		std::cout << arg << "\n";

#elif defined (WIN32) || defined (_WIN32)
		if (color == RESET) {
			system("COLOR F");
			std::cout << arg << "\n"; 
		}
		else {
			system((std::string("COLOR ") + std::to_string(color)).c_str());
			std::cout << arg << "\n";

		}


#endif
	}

/*******************************************************************************************************************************************************/


/*ProgramStatus class methods*/


	ProgramStatus::ProgramStatus(){
		_idCurrentImage = -1; 

	}

	ProgramStatus::~ProgramStatus(){


	}



/*******************************************************************************************************************************************************/

}
