#ifndef IMAGEIMPORTER_H
#define IMAGEIMPORTER_H

#include "Model.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
namespace maptomix{


class ImageImporter: public Model

{

	public:


		static ImageImporter* getInstance();
		static void close();
		SDL_Surface* load_image(const char* file) ;


	private:
		
		ImageImporter(); 
		~ImageImporter(); 
		static ImageImporter* instance;
		




};













}










#endif
