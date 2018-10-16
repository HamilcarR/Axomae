#include "../includes/ImageImporter.h"
#include <iostream>
#include <assert.h>



using namespace std;
namespace axomae {

ImageImporter* ImageImporter::instance = nullptr;


ImageImporter::ImageImporter()
	
{

}

ImageImporter::~ImageImporter(){
	
		SDL_FreeSurface(surf);
}

/*************************************************************************************************************/

ImageImporter* ImageImporter::getInstance(){
	if(instance == nullptr)
		instance = new ImageImporter(); 
	return instance;

}


/**************************************************************************************************************/


void ImageImporter::close(){
	delete instance;

}



/**************************************************************************************************************/


SDL_Surface* ImageImporter::load_image(const char* file){
        surf = IMG_Load(file);
	if(!surf)
		cout<<"Image loading problem : " << IMG_GetError()<<endl;
	
	return surf;


}






/**************************************************************************************************************/


void ImageImporter::save_image(SDL_Surface* surface,const char* filename){
	SDL_SaveBMP(surface,filename);


}














}









