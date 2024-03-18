#include "API.h"
#include "CmdArgs.h"
#include "GUIWindow.h"
#include "GenericException.h"
#include "ImageImporter.h"
#include <boost/program_options.hpp>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace axomae;

static void init_graphics() {
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    std::cout << "SDL_Init problem : " << SDL_GetError() << "\n";
  }
  if (!IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG)) {
    std::cout << "IMG init problem : " << IMG_GetError() << "\n";
  }
}

static void cleanup() {
  IMG_Quit();
  SDL_Quit();
}

static void sigsegv_handler(int signal) {
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

static int exception_cleanup(const char *except_error) {
  std::cerr << except_error << "\n";
  cleanup();
  return EXIT_FAILURE;
}

int main(int argv, char **argc) {
  signal(SIGSEGV, sigsegv_handler);
  controller::cmd::API api(argv, argc);
  controller::cmd::ProgramOptionsManager options_manager(&api);
  api.configureDefault();
  init_graphics();
  if (argv >= 2) {
    try {
      options_manager.processArgs(argv, argc);
    } catch (const boost::program_options::error &e) {
      return exception_cleanup(e.what());
    } catch (const exception::CatastrophicFailureException &e) {
      return exception_cleanup(e.what());
    }
    api.configure();
    const ApplicationConfig &configuration = api.getConfig();
    if (configuration.flag & CONF_USE_EDITOR) {
      QApplication app(argv, argc);
      controller::Controller win;
      win.setApplicationConfig(&configuration);
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
    api.configureDefault();
    const ApplicationConfig &configuration = api.getConfig();
    win.setApplicationConfig(&configuration);
    win.show();
    int ret = app.exec();
    cleanup();
    return ret;
  }
}
