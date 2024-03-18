#ifndef GENERICEXCEPTION_H
#define GENERICEXCEPTION_H
#include <stdexcept>

namespace exception {
  class ExceptionData {
   public:
    ExceptionData(const std::string &error_string);

    virtual ~ExceptionData() = default;

    ExceptionData(const ExceptionData &copy);

    ExceptionData(ExceptionData &&move) noexcept;

    ExceptionData &operator=(const ExceptionData &copy);

    ExceptionData &operator=(ExceptionData &&move) noexcept;

    virtual void saveErrorString(const std::string &string) {
      this_error_string += string;
      this_error_string += "\n";
    }

    virtual void saveErrorString(const char *string) {
      std::string str(string);
      saveErrorString(str);
    }

   protected:
    std::string this_error_string;
  };

  class GenericException : public std::exception, protected ExceptionData {
   public:
    GenericException() : std::exception(), ExceptionData(std::string("The program has encountered an exception : \n")) {}
    [[nodiscard]] const char *what() const noexcept override { return this_error_string.c_str(); }
  };

  class InvalidArgumentException : public std::invalid_argument, protected ExceptionData {
   public:
    explicit InvalidArgumentException(const std::string &error_string)
        : std::invalid_argument(error_string), ExceptionData(std::string("The program has encoutered an exception : \n ")) {}
    [[nodiscard]] const char *what() const noexcept override { return this_error_string.c_str(); }
  };

  class CatastrophicFailureException : public std::exception, protected ExceptionData {
   public:
    CatastrophicFailureException()
        : std::exception(), ExceptionData(std::string("The program has encoutered a critical exception and will now shut down :\n")) {}
    [[nodiscard]] const char *what() const noexcept override { return this_error_string.c_str(); }
  };

}  // namespace exception
#endif