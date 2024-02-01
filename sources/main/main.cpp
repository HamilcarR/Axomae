#include "CmdArgs.h"
#include "GUIWindow.h"
#include "ImageImporter.h"
#include <boost/program_options.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <signal.h>
#include <string>
using namespace std;
using namespace axomae;

void init_graphics() {
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    cout << "SDL_Init problem : " << SDL_GetError() << endl;
  }
  if (!IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG)) {

    cout << "IMG init problem : " << IMG_GetError() << endl;
  }
}

void cleanup() {
  IMG_Quit();
  SDL_Quit();
}

void sigsegv_handler(int signal) {
  try {
    LOG("Application crash", LogLevel::CRITICAL);
    LOGFLUSH();
    // Generate stack here
    cleanup();
  } catch (const std::exception &e) {
    std::cerr << e.what();
  }
  abort();
}

int main(int argv, char **argc) {
  signal(SIGSEGV, sigsegv_handler);
  ApplicationConfig configuration;
  controller::cmd::API api(argv, argc);
  controller::cmd::ProgramOptionsManager options_manager(api);
  LoggerConfigDataStruct log_struct = configuration.generateDefaultLoggerConfigDataStruct();
  api.configureDefault();
  init_graphics();
  if (argv >= 2) {
    try {
      options_manager.processArgs(argv, argc);
    } catch (const boost::program_options::error &e) {
      std::cerr << e.what();
      cleanup();
      return EXIT_FAILURE;
    }
    configuration = api.getConfig();
    LoggerConfigDataStruct log_conf = configuration.generateLoggerConfigDataStruct();
    api.configure();
    if (configuration.getGuiState()) {
      QApplication app(argv, argc);
      controller::Controller win;
      win.setApplicationConfig(configuration);
      win.show();
      int ret = app.exec();
      cleanup();
      return ret;
    } else {
      cleanup();
      return EXIT_SUCCESS;
    }
  } else {
    QApplication app(argv, argc);
    controller::Controller win;
    win.setApplicationConfig(configuration);
    win.show();
    int ret = app.exec();
    cleanup();
    return ret;
  }
}
