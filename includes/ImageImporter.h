#ifndef IMAGEIMPORTER_H
#define IMAGEIMPORTER_H

#include "Model.h"
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
namespace axioma{


class ImageImporter: public Model

{

	public:


		static ImageImporter* getInstance();
		static void close();
		SDL_Surface* load_image(const char* file) ;
		static void save_image(SDL_Surface* surface,const char* filename);

	private:
		
		ImageImporter(); 
		~ImageImporter();
	        SDL_Surface* surf;	
		static ImageImporter* instance;
		




};













}










#endif
