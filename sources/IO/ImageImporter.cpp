#include "ImageImporter.h"
#include "Logger.h"
#include <SDL.h>
#include <SDL_image.h>
#include <assert.h>
#include <iostream>

using namespace std;
namespace IO {

  ImageImporter *ImageImporter::instance = nullptr;

  ImageImporter::ImageImporter() = default;

  ImageImporter::~ImageImporter() { SDL_FreeSurface(surf); }

  /*************************************************************************************************************/

  ImageImporter *ImageImporter::getInstance() {
    if (instance == nullptr)
      instance = new ImageImporter();
    return instance;
  }
  /**************************************************************************************************************/
  void ImageImporter::close() { delete instance; }

  /**************************************************************************************************************/

  SDL_Surface *ImageImporter::load_image(const char *file) {
    surf = IMG_Load(file);
    if (!surf)
      LOG("Image loading problem : " + std::string(IMG_GetError()), LogLevel::ERROR);
    return surf;
  }

  /**************************************************************************************************************/
  void ImageImporter::save_image(SDL_Surface *surface, const char *filename) { SDL_SaveBMP(surface, filename); }

}  // namespace IO
