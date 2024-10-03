#ifndef LOADERSHAREDEXCEPTIONS_H
#define LOADERSHAREDEXCEPTIONS_H
#include "internal/common/exception/GenericException.h"

namespace exception {
  class LoadFilePathException : public exception::GenericException {
   public:
    explicit LoadFilePathException(const std::string &path) {
      GenericException::saveErrorString(std::string("Failed processing path for image : ") + path);
    }
  };

}  // namespace exception

#endif  // LOADERSHAREDEXCEPTIONS_H
