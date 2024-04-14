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

#define LOG(message, level) log_functions::log_message(message, level, __FILE__, __func__, __LINE__)
#define LOGFL(message, level, file, function, line) log_functions::log_message(message, level, file, function, line)
#define LOGS(message) log_functions::log_message(message)
#define LOGFLUSH() log_functions::log_flush()
#define LOGCONFIG(config) log_functions::log_configure(config)
#define LOGDISABLE() log_functions::log_disable()
#define LOGENABLE() log_functions::log_enable()

namespace LogLevel {
  enum LOGENUMTYPE : unsigned { INFO, GLINFO, WARNING, ERROR, CRITICAL, DEBUG };
}  // End namespace LogLevel

struct LoggerConfigDataStruct {
  std::shared_ptr<std::ostream> write_destination;
  LogLevel::LOGENUMTYPE log_level;
  std::string log_filters;
  bool enable_logging;
};

namespace log_functions {
  void log_message(const char *message, LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line);
  void log_message(const std::string &message, LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line);
  void log_message(const char *message);
  void log_message(const std::string &message);
  void log_flush();
  void log_configure(const LoggerConfigDataStruct &config);
  void log_disable();
  void log_enable();
}  // namespace log_functions

/*****************************************************************************************************************************************************************************/
class AbstractLogger : public ILockable {
 public:
  virtual ~AbstractLogger() = default;
  virtual void print() const = 0;
};

#endif