#ifndef TERMOPT_H
#define TERMOPT_H
#include "ImageImporter.h"
#include "ImageManager.h"
#include "Renderer.h"
#include "Window.h"
#include <map>
#include <memory>
#include <queue>
#include <stack>
#if defined(WIN32) || defined(_WIN32)
#  include <Windows.h>
#endif

namespace axomae {

  enum : unsigned { PROMPT0 = 0, PROMPT1 = 1, PROMPT2 = 2, PROMPT3 = 3 };

  static const char *prompt[] = {">", " :", "=:", ">>"

  };

  static const char *command[] = {
      "window",     // create a new window
      "normalMap",  // compute normal map
      "heightMap",  // compute height map
      "dudv",       // compute DUDV
      "save",       // save image
      "contrast",   // set contrast
      "exit",       // exit the program
      "render",     // render an object on a OpenGL window
      "load"        // load an image
  };

  enum : unsigned {
    CHK_CURRENT_IMG = 11,
    SELECT = 10,
    LISTIDS = 9,
    LOAD = 8,
    RENDER = 7,
    EXIT = 6,
    CONTRAST = 5,
    SAVE = 4,
    DUDV = 3,
    HMAP = 2,
    NMAP = 1,
    WIN = 0
  };
  enum : unsigned { WIN_ARGS = 3 };

#ifdef __unix__
  enum : unsigned { RED = 0, BLUE = 1, GREEN = 2, YELLOW = 3, RESET = 4 };

  static const char *colors[] = {"\033[31m", "\033[34m", "\033[32m", "\033[33m", "\033[0m"

  };

#elif defined(WIN32) || defined(_WIN32)
  enum : unsigned { RED = 4, BLUE = 1, GREEN = 2, PURPLE = 5, YELLOW = 6, RESET = 10 };

  /*Color Codes:
  0 = Black
  1 = Blue
  2 = Green
  3 = Aqua
  4 = Red
  5 = Purple
  6 = Yellow
  7 = White
  8 = Gray
  9 = Light Blue
  A = Light Green
  B = Light Aqua
  C = Light Red
  D = Light Purple
  E = Light Yellow
  F = Bright White
  */

#endif

  /*******************************************************************************************************************************************************/

  static void print(std::string arg, int8_t color, int8_t p = -1) {
#ifdef __unix__
    std::cout << colors[color] << "\n";
    std::cout << ((p >= 0 && p < 4) ? std::string(prompt[p]) + " " : "") << arg << "\n";

#elif defined(WIN32) || defined(_WIN32)
    HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    if (color == RESET) {
      SetConsoleTextAttribute(console, 15);
      std::cout << "\n" << ((p >= 0 && p < 4) ? std::string(prompt[p]) + " " : "") << arg;
    } else {
      SetConsoleTextAttribute(console, color);
      std::cout << "\n" << ((p >= 0 && p < 4) ? std::string(prompt[p]) + " " : "") << arg;
    }

#endif
  }

  static void print(const char *text, int8_t color, int8_t prompt) {
    print(std::string(text), color, prompt);
  }
  static void print(char *text, int8_t color, int8_t prompt) {
    print(std::string(text), color, prompt);
  }

  static void print(const char *text, int8_t color) {
    print(std::string(text), color);
  }
  static void print(char *text, int8_t color) {
    print(std::string(text), color);
  }

  typedef std::vector<std::pair<std::shared_ptr<Window>, SDL_Event>> WindowsStack;
  typedef std::vector<std::pair<SDL_Surface *, std::string>> ImagesStack;

  class ProgramStatus {
   public:
    static ProgramStatus *getInstance() {
      if (instance == nullptr)
        instance = new ProgramStatus();
      return instance;
    }
    static void Quit() {
      if (instance != nullptr)
        delete instance;
      instance = nullptr;
    }

    void process_command(std::string user_input);
    std::vector<std::pair<SDL_Surface *, std::string>> getImages() {
      return images;
    }
    std::vector<std::pair<std::shared_ptr<Window>, SDL_Event>> getWindows() {
      return windows;
    }
    int getCurrentImageId() {
      return _idCurrentImage;
    }
    void setDisplayNULL() {
      _display_window = nullptr;
    }
    Window *getDisplay() {
      return _display_window;
    }
    void setDisplay(Window *d) {
      _display_window = d;
    }
    void exit();
    void setLoop(bool);
    bool getLoop();

   private:
    /*functions*/
    ProgramStatus();
    ~ProgramStatus();
    void create_window(int, int, const char *);

    /*attributes*/

    std::vector<std::pair<SDL_Surface *, std::string>> images;
    std::map<std::thread, std::pair<std::shared_ptr<Window>, SDL_Event>> thread_pool;
    std::vector<std::pair<std::shared_ptr<Window>, SDL_Event>> windows;
    std::vector<std::thread> _threads; /*std::move used */
    std::shared_ptr<Renderer> _renderer;
    bool loop;
    int _idCurrentImage;
    Window *_display_window;
    static ProgramStatus *instance;
    bool exited;
  };

}  // namespace axomae

#endif
