#include "API.h"
#include "Logger.h"
namespace controller::cmd {
  const int DEFAULT_UV_RES_X = 700;
  const int DEFAULT_UV_RES_Y = 700;

  void API::configureDefault() {
    LoggerConfigDataStruct conf = config.generateDefaultLoggerConfigDataStruct();
    conf.enable_logging = config.flag & CONF_ENABLE_LOGS;
    config.setUvEditorResolutionWidth(DEFAULT_UV_RES_X);
    config.setUvEditorResolutionHeight(DEFAULT_UV_RES_Y);
    config.initializeThreadPool();
    LOGCONFIG(conf);
  }
  void API::configure() {
    configureDefault();
    /* Init logger */
    LoggerConfigDataStruct conf = config.generateLoggerConfigDataStruct();
    std::shared_ptr<std::ostream> out(&std::cout, [](std::ostream *) {});
    conf.write_destination = out;
    conf.enable_logging = config.flag & CONF_ENABLE_LOGS;
    LOGCONFIG(conf);
    /* Other configs */
  }
}  // namespace controller::cmd