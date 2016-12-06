#include "../includes/ImageImporter.h"
#include "../includes/TerminalOpt.h"
#include <regex>
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <algorithm>
namespace maptomix{
	
	const int STRING_SIZE = 256 ; 

	ProgramStatus *ProgramStatus::instance = nullptr; 
	

	static const std::regex command_regex[]={
		std::regex("window [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),	
		std::regex("nmap [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("hmap [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("dudv [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("save [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("contrast [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("exit [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("render [0-9]+ [0-9]+ [a-z]+",std::regex_constants::icase),
		std::regex("load [a-z]+.[a-z0-9]+",std::regex_constants::icase)


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

		if(save){
			/*create a "state" class with pointers to every elements: Save the image of the window if a window exists*/
		std::cout << "dshsds" <<std::endl; 

		}
		else if(load){
			Validation v = validate_command_load(user_input);
			if (v.validated)
			{
				std::cout << "File : " << v.command_arguments[0] << " loading..." << "\n" ; 
				ImageImporter *instance = ImageImporter::getInstance(); 
				SDL_Surface* im = instance->load_image(static_cast<const char*>(v.command_arguments[0].c_str()));
				if(im)
				images.push_back(im);				
				


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
				display = std::unique_ptr<Window>(new Window(w,h,window_name.c_str()));
				display->setEvent(event); 
				if(images.size()!=0){
					bool loop = true; 
					while(loop){
						while(SDL_PollEvent(&event)){
							if(event.type == SDL_QUIT)
								loop = false; 

						}

					display->display_image(images[0]); 
					}
				}	

			std::puts("Exiting...\n"); 
			display.reset(nullptr); 
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
		
		else if(closew ){
			std::puts("Exiting...\n"); 
			display.reset(); 

		}

	
	}






/*******************************************************************************************************************************************************/


/*ProgramStatus class methods*/


	ProgramStatus::ProgramStatus(){


	}

	ProgramStatus::~ProgramStatus(){


	}



/*******************************************************************************************************************************************************/

}
