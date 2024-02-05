#ifndef LOADERSHAREDEXCEPTIONS_H
#define LOADERSHAREDEXCEPTIONS_H
#include "GenericException.h"

namespace IO {
  namespace exception {
    class LoadFilePathException : public GenericException {
     public:
      explicit LoadFilePathException(const std::string &path) {
        GenericException::saveErrorString(std::string("Failed processing path for image : ") + path);
      }
    };
  }  // namespace exception

}  // namespace IO

#endif  // LOADERSHAREDEXCEPTIONS_H
