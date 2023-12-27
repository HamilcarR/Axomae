#ifndef GENERICEXCEPTION_H
#define GENERICEXCEPTION_H
#include <stdexcept>

class GenericException : virtual public std::exception {
 public:
  GenericException() : std::exception() { this_error_string = std::string("The program has encountered an exception : \n"); }

  const char *what() const noexcept override { return this_error_string.c_str(); }

  virtual void saveErrorString(const std::string &string) {
    this_error_string += string;
    this_error_string += "\n";
  }

  virtual void saveErrorString(const char *string) {
    std::string str(string);
    saveErrorString(str);
  }

 private:
  std::string this_error_string;
};

#endif