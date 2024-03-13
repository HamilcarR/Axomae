#include "Logger.h"
#include "GenericException.h"
#include "Mutex.h"
#include <map>
#include <unordered_map>
#define LOG2STR(level) level_str[level]
#define STR2LOG(str) level_str.find(str)

std::unordered_map<LogLevel::LOGENUMTYPE, std::string> level_str = {{LogLevel::CRITICAL, "CRITICAL"},
                                                                    {LogLevel::DEBUG, "DEBUG"},
                                                                    {LogLevel::ERROR, "ERROR"},
                                                                    {LogLevel::INFO, "INFO"},
                                                                    {LogLevel::WARNING, "WARNING"},
                                                                    {LogLevel::GLINFO, "OPENGL INFO"}};

static std::string getFormatedLog(
    const std::string &message, const std::string &filename, const std::string &function, const unsigned line, LogLevel::LOGENUMTYPE level) {
  std::string head = filename + ";" + function + "();" + std::to_string(line) + ";" + LOG2STR(level) + "==> " + message;
  return head;
}

/***************************************************************************************************************************************************************/

class LoggerOutputStreamException : public exception::GenericException {
 public:
  LoggerOutputStreamException() : GenericException() { saveErrorString("The output stream for the application logger system is undefined !"); }
  virtual ~LoggerOutputStreamException() {}
};

/***************************************************************************************************************************************************************/
class LogLine {
 public:
  LogLine(const std::string &_message,
          const LogLevel::LOGENUMTYPE log_level = LogLevel::INFO,
          const char *_file = "",
          const char *_function = "",
          const unsigned _line = 0) {
    message = _message;
    level = log_level;
    file = _file;
    line = _line;
    function = _function;
  }
  const std::string &getMessage() { return message; }

  const std::string &getLoggedFile() { return file; }

  const std::string &getFunctionName() { return function; }

  unsigned getLine() { return line; }

  LogLevel::LOGENUMTYPE &getLogLevel() { return level; }

  std::string getFormattedLog() const { return getFormatedLog(message, file, function, line, level); }

 private:
  std::string message;
  LogLevel::LOGENUMTYPE level;
  std::string file{};
  std::string function{};
  unsigned line;
};
/***************************************************************************************************************************************************************/
class Logger : protected AbstractLogger {
 public:
  Logger() { enabled = true; }

  void logMessage(const std::string &message, LogLevel::LOGENUMTYPE log_level, const char *file, const char *function, unsigned line) {
    Mutex::Lock lock(mutex);
    LogLine elem(message, log_level, file, function, line);
    log_buffer.push_back(elem);
    if (enabled)
      print();
  }

  void logMessage(const char *message, LogLevel::LOGENUMTYPE log_level, const char *file, const char *function, unsigned line) {
    logMessage(std::string(message), log_level, file, function, line);
  }

  void logMessage(const char *message) {
    if (enabled)
      *out << level_str[LogLevel::INFO] << message << "\n";
  }

  void print() const {
    if (out) {
      auto last_log = log_buffer.back();
      *out << last_log.getFormattedLog() << "\n";
    }
  }

  void flush() {
    if (!out)
      throw LoggerOutputStreamException();
    if (!stdout_logging) {
      Mutex::Lock lock(mutex);
      for (LogLine elem : log_buffer)
        *out << getFormatedLog(elem.getMessage(), elem.getLoggedFile(), elem.getFunctionName(), elem.getLine(), elem.getLogLevel()) << "\n";
    }
  }

  void setLogSystemConfig(const LoggerConfigDataStruct &config) {
    Mutex::Lock lock(mutex);
    filters = config.log_filters;
    priority = config.log_level;
    out = config.write_destination;
    enabled = config.enable_logging;
  }

  void setPriority(LogLevel::LOGENUMTYPE _priority) { priority = _priority; }

  void setLoggingStdout() { stdout_logging = true; }

  std::ostream &outstm() { return *out; }

  void loggerState(bool enabled_) { enabled = enabled_; }

 protected:
  std::vector<LogLine> log_buffer{};
  bool stdout_logging{};
  bool enabled{};
  LogLevel::LOGENUMTYPE priority;
  std::shared_ptr<std::ostream> out;
  std::string filters{};
};
/***************************************************************************************************************************************************************/

static Logger logger_global;

namespace log_functions {
  void log_message(std::string message, const LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line) {
    logger_global.logMessage(message, level, file, function, line);
  }

  void log_message(const char *message, const LogLevel::LOGENUMTYPE level, const char *file, const char *function, unsigned int line) {
    log_message(std::string(message), level, file, function, line);
  }

  void log_message(const char *message) { logger_global.logMessage(message); }

  void log_flush() { logger_global.flush(); }

  void log_configure(const LoggerConfigDataStruct &conf) { logger_global.setLogSystemConfig(conf); }

  void log_enable() { logger_global.loggerState(true); }

  void log_disable() { logger_global.loggerState(false); }
}  // namespace log_functions
