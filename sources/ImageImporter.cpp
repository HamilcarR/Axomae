#include "../includes/ImageImporter.h"
#include <iostream>
#include <assert.h>



using namespace std;
namespace maptomix {

ImageImporter* ImageImporter::instance = nullptr;


ImageImporter::ImageImporter():maptomix::Model()
	
{


}

ImageImporter::~ImageImporter(){


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
	SDL_Surface* surf = IMG_Load(file);
	if(!surf)
		cout<<"Img load problem : " << IMG_GetError()<<endl;
	assert(surf!=nullptr);
	return surf;


}























}









