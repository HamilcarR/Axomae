#ifndef GENERICEXCEPTION_H
#define GENERICEXCEPTION_H
#include "class_macros.h"
#include <stdexcept>
namespace exception {
  enum SEVERITY : int { QUERY, INFO, WARNING, CRITICAL };
  class ExceptionData {
   protected:
    std::string this_error_string;
    SEVERITY severity;

   public:
    CLASS_VCM(ExceptionData)

    explicit ExceptionData(const std::string &error_string, SEVERITY severity_ = INFO) : this_error_string(error_string) { severity = severity_; }
    void saveErrorString(const std::string &string) {
      this_error_string += string;
      this_error_string += "\n";
    }
    void saveErrorString(const char *string) {
      std::string str(string);
      saveErrorString(str);
    }
    void setSeverity(SEVERITY n) { severity = n; }
    [[nodiscard]] SEVERITY getSeverity() const { return severity; }
    [[nodiscard]] const std::string &getErrorMessage() const { return this_error_string; }
  };

  class GenericException : public std::exception, public ExceptionData {
   public:
    GenericException() : std::exception(), ExceptionData(std::string("The program has encountered an exception : \n")) {}
    GenericException(const std::string &message, SEVERITY severity) : std::exception(), ExceptionData(message, severity) {}
    [[nodiscard]] const char *what() const noexcept override { return this_error_string.c_str(); }
  };

  class InvalidArgumentException : public std::invalid_argument, public ExceptionData {
   public:
    explicit InvalidArgumentException(const std::string &error_string)
        : std::invalid_argument(error_string), ExceptionData(std::string("The program has encoutered an exception : \n ")) {}
    InvalidArgumentException(const std::string &error_string, SEVERITY severity)
        : std::invalid_argument(error_string), ExceptionData(error_string, severity) {}
    [[nodiscard]] const char *what() const noexcept override { return this_error_string.c_str(); }
  };

  class CatastrophicFailureException : public std::exception, public ExceptionData {
   public:
    CatastrophicFailureException()
        : std::exception(), ExceptionData(std::string("The program has encoutered a critical exception and will now shut down \n"), CRITICAL) {}
    [[nodiscard]] const char *what() const noexcept override { return this_error_string.c_str(); }
  };

}  // namespace exception
#endif