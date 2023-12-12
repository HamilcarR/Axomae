#ifndef LOGGER_H
#define LOGGER_H
#include "Axomae_macros.h"
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

namespace LogLevel {
  enum LOGENUMTYPE : unsigned { INFO, GLINFO, WARNING, ERROR, CRITICAL, DEBUG };
};  // End namespace LogLevel

struct LoggerConfigDataStruct {
  std::shared_ptr<std::ostream> write_destination;
  LogLevel::LOGENUMTYPE log_level;
  std::string log_filters;
};

namespace LogFunctions {
  void log_message(const char *message, const LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line);
  void log_message(std::string message, const LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line);
  void log_flush();
  void log_configure(const LoggerConfigDataStruct &config);
};  // namespace LogFunctions
/*****************************************************************************************************************************************************************************/
class AbstractLogger {
 public:
  AbstractLogger();
  virtual ~AbstractLogger();
  virtual void print() const = 0;

 protected:
  mutable Mutex mutex;
};
/*****************************************************************************************************************************************************************************/
class LogLine {
 public:
  LogLine(const std::string &message, const LogLevel::LOGENUMTYPE log_level = LogLevel::INFO, const char *file = "", const char *function = "", const unsigned line = 0);
  virtual ~LogLine();
  const std::string &getMessage() {
    return message;
  }
  const std::string &getLoggedFile() {
    return file;
  }
  const std::string &getFunctionName() {
    return function;
  }
  unsigned getLine() {
    return line;
  }
  LogLevel::LOGENUMTYPE &getLogLevel() {
    return level;
  }
  std::string getFormattedLog() const;

 private:
  std::string message;
  LogLevel::LOGENUMTYPE level;
  std::string file;
  std::string function;
  unsigned line;
};

/*****************************************************************************************************************************************************************************/
class Logger : virtual public AbstractLogger {
 public:
  Logger();
  virtual ~Logger();
  virtual void print() const;
  void logMessage(const std::string &message, LogLevel::LOGENUMTYPE log_level, const char *filename, const char *function, unsigned line);
  void logMessage(const char *message, LogLevel::LOGENUMTYPE log_level, const char *filename, const char *function, unsigned line);
  void flush();
  void setPriority(LogLevel::LOGENUMTYPE _priority) {
    priority = _priority;
  }
  void setLoggingStdout() {
    stdout_logging = true;
  }
  void setLogSystemConfig(const LoggerConfigDataStruct &conf);

 protected:
  std::vector<LogLine> log_buffer;
  bool stdout_logging;
  LogLevel::LOGENUMTYPE priority;
  std::shared_ptr<std::ostream> out;
  std::string filters;
};

#endif