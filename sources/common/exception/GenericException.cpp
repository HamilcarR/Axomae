#include "GenericException.h"

namespace exception {

  ExceptionData::ExceptionData(const std::string &error_string) : this_error_string(error_string) {}

  ExceptionData::ExceptionData(const ExceptionData &copy) : this_error_string(copy.this_error_string) {}

  ExceptionData::ExceptionData(ExceptionData &&move) noexcept : this_error_string(std::move(move.this_error_string)) { move.this_error_string = ""; }

  ExceptionData &ExceptionData::operator=(ExceptionData &&move) noexcept {
    this_error_string = std::move(move.this_error_string);
    move.this_error_string = "";
    return *this;
  }
  ExceptionData &ExceptionData::operator=(const ExceptionData &copy) {
    this_error_string = copy.this_error_string;
    return *this;
  }

}  // namespace exception
