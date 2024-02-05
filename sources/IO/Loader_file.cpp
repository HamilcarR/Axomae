#include "Loader.h"
#include "LoaderSharedExceptions.h"
#include <fstream>

namespace IO {

  std::string Loader::loadTextFile(const char *filename) {
    std::ifstream stream(filename);
    if (!stream.is_open())
      throw exception::LoadFilePathException(filename);
    std::string buffer;
    std::string str_text;
    while (getline(stream, buffer))
      str_text = str_text + buffer + "\n";
    stream.close();
    return str_text;
  }

}  // namespace IO