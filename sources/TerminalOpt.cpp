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
			if(c1 && c2 ){
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


	void process_command(std::string user_input){
		/*all inputs*/
		std::regex matching_save (command[SAVE] , std::regex_constants::icase) ; 
		std::regex matching_load(command[LOAD],std::regex_constants::icase);
		std::regex matching_window(command[WIN] , std::regex_constants::icase); 
		std::regex matching_normalmap(command[NMAP] , std::regex_constants::icase); 
		std::regex matching_heightmap(command[HMAP] , std::regex_constants::icase); 
		std::regex matching_contrast(command[CONTRAST],std::regex_constants::icase); 
		std::regex matching_render(command[RENDER],std::regex_constants::icase); 
		std::regex matching_dudv(command[DUDV],std::regex_constants::icase);

		/*does it match ?*/

		bool save = std::regex_search(user_input,matching_save);
		bool load= std::regex_search(user_input,matching_load);
		bool window = std::regex_search(user_input,matching_window);
		bool normalmap = std::regex_search(user_input,matching_normalmap);
		bool heightmap = std::regex_search(user_input,matching_heightmap);
		bool contrast = std::regex_search(user_input,matching_contrast);
		bool render = std::regex_search(user_input,matching_render);
		bool dudv = std::regex_search(user_input,matching_dudv);


		if(save){
			/*create a "state" class with pointers to every elements: Save the image of the window if a window exists*/


		}
		else if(load){

		}
		else if(window){
			Validation v = validate_command_window(user_input); 
			if(v.validated)
			{
				for(std::string c : v.command_arguments)
					std::cout<<c<<"\n";
				//TODO create window 
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
		

		

	}













}
