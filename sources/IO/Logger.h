#ifndef LOGGER_H
#define LOGGER_H
#include "Axomae_macros.h"
#include "ILockable.h"
#include "Mutex.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#define LOG(message, level) LogFunctions::log_message(message, level, __FILE__, __func__, __LINE__)
#define LOGFLUSH() LogFunctions::log_flush()
#define LOGCONFIG(config) LogFunctions::log_configure(config)
#define LOGDISABLE() LogFunctions::log_disable()
#define LOGENABLE() LogFunctions::log_enable()
namespace LogLevel {
  enum LOGENUMTYPE : unsigned { INFO, GLINFO, WARNING, ERROR, CRITICAL, DEBUG };
}  // End namespace LogLevel

struct LoggerConfigDataStruct {
  std::shared_ptr<std::ostream> write_destination;
  LogLevel::LOGENUMTYPE log_level;
  std::string log_filters;
  bool enable_logging;
};

namespace LogFunctions {
  void log_message(const char *message, const LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line);
  void log_message(std::string message, const LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line);
  void log_message(const char *message);
  void log_flush();
  void log_configure(const LoggerConfigDataStruct &config);
  void log_disable();
  void log_enable();
}  // namespace LogFunctions

/*****************************************************************************************************************************************************************************/
class AbstractLogger : public ILockable {
 public:
  virtual void print() const = 0;
};

#endif