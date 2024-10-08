#ifndef IMAGEIMPORTER_H
#define IMAGEIMPORTER_H

class SDL_Surface;
// TODO : Old code , move to Loader_image.cpp
namespace IO {

  class ImageImporter {
   private:
    SDL_Surface *surf{};
    static ImageImporter *instance;

   public:
    static ImageImporter *getInstance();
    static void close();
    SDL_Surface *load_image(const char *file);
    static void save_image(SDL_Surface *surface, const char *filename);

   private:
    ImageImporter();
    ~ImageImporter();
  };

}  // namespace IO

#endif
