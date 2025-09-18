#include "GUIWindow.h"
#include "cmd/API.h"
#include "cmd/ProgramOptionsManager.h"
#include "internal/common/exception/GenericException.h"
#include <boost/program_options.hpp>
#include <boost/stacktrace.hpp>
#include <csignal>
#include <cstdlib>
#include <iostream>
using namespace axomae;

static void init_graphics() {}

static void cleanup() {}

static void sigsegv_handler(int signal) {
  LOG("Application crash", LogLevel::CRITICAL);
  LOGFLUSH();
  std::cerr << boost::stacktrace::stacktrace();
  cleanup();
}

static int exception_cleanup(const char *except_error) {
  std::cerr << except_error << "\n";
  cleanup();
  return EXIT_FAILURE;
}

int main(int argv, char **argc) {
  try {
    std::thread::id main_thread = std::this_thread::get_id();
    signal(SIGSEGV, sigsegv_handler);
    controller::cmd::API api(argv, argc, main_thread);
    controller::cmd::ProgramOptionsManager options_manager(&api);
    api.configureDefault();
    init_graphics();
    if (argv >= 2) {
      options_manager.processArgs(argv, argc);
      api.configure();
      std::unique_ptr<ApplicationConfig> configuration = api.getConfig();
      if (configuration->flag & CONF_USE_EDITOR) {
        QApplication app(argv, argc);
        controller::Controller win(std::move(configuration));
        win.show();
        int ret = app.exec();
        cleanup();
        return ret;
      }
      cleanup();
      return EXIT_SUCCESS;
    }
    QApplication app(argv, argc);
    api.configureDefault();
    std::unique_ptr<ApplicationConfig> configuration = api.getConfig();
    controller::Controller win(std::move(configuration));
    win.show();
    int ret = app.exec();
    cleanup();
    return ret;
  } catch (const boost::program_options::error &e) {
    return exception_cleanup(e.what());
  } catch (const exception::CatastrophicFailureException &e) {
    return exception_cleanup(e.what());
  } catch (const std::exception &e) {
    std::cerr << e.what();
  }
}
