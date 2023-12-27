#ifndef WINDOW_H
#define WINDOW_H
#include "image_utils.h"
#include <SDL.h>
#include <SDL_image.h>
#include <string>
#include <thread>

namespace axomae {

  struct thread_data;
  /* Stand alone window*/
  class Window {
   public:
    Window(const int width, const int height, const char *name);
    ~Window();
    void display_image(SDL_Surface *image);
    void setEvent(SDL_Event &ev) { event = ev; }
    SDL_Event &getEvent() { return event; }
    SDL_Renderer *getRenderer() { return renderer; };
    int getHeight() { return height; }
    int getWidth() { return width; }
    void cleanUp();

   private:
    int width;
    int height;
    std::string name;
    SDL_Window *m_window;
    SDL_Renderer *renderer;
    SDL_Event event;
    SDL_Texture *texture;
    bool free_surface_texture;
    //	std::thread t_sdl;
  };

}  // namespace axomae

#endif
