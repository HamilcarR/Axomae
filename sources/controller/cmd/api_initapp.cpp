#include "API.h"
#include "Logger.h"
namespace controller::cmd {

  void API::configureDefault() {
    LoggerConfigDataStruct conf = config.generateDefaultLoggerConfigDataStruct();
    LOGCONFIG(conf);
  }

  void API::configure() {
    /* Init logger */
    LoggerConfigDataStruct conf = config.generateLoggerConfigDataStruct();
    std::shared_ptr<std::ostream> out(&std::cout, [](std::ostream *) {});
    conf.write_destination = out;
    LOGCONFIG(conf);

    /* Other configs */
  }
}  // namespace controller::cmd