#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <regex>
#include <signal.h>
#include <string>
#include <thread>

#include "GUIWindow.h"
#include "ImageImporter.h"
#include "ImageManager.h"
#include "TerminalOpt.h"
#include "Window.h"

using namespace std;
using namespace axomae;
using namespace controller;

void init_api() {
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    cout << "SDL_Init problem : " << SDL_GetError() << endl;
  }
  if (!IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG)) {

    cout << "IMG init problem : " << IMG_GetError() << endl;
  }
}

void sigsegv_handler(int signal) {
  try {
    LOG("Application crash", LogLevel::CRITICAL);
    LOGFLUSH();
  } catch (const std::exception &e) {
    std::cerr << e.what();
  }
  abort();
}

void quit_api() {
  IMG_Quit();
  SDL_Quit();
}

int main(int argv, char **argc) {
  signal(SIGSEGV, sigsegv_handler);
  init_api();
  if (argv >= 2) {
    QApplication app(argv, argc);
    Controller win;
    /* For future cmd arguments */
    std::string param_string = "";
    for (int i = 1; i < argv; i++)
      param_string += argc[i] + std::string(" ");
    win.setApplicationConfig(param_string);
    win.show();
    return app.exec();
  } else {
    QApplication app(argv, argc);
    Controller win;
    win.show();
    return app.exec();
  }
  quit_api();
  return EXIT_SUCCESS;
}
