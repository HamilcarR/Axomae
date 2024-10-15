#ifndef AXOMAE_UTILS_H
#define AXOMAE_UTILS_H
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
namespace utils {
  namespace string {
    inline std::vector<std::string> tokenize(const std::string &str, char delim) {
      std::istringstream strstrm(str);
      std::string token;
      std::vector<std::string> ret;
      while (std::getline(strstrm, token, delim)) {
        ret.push_back(token);
      }
      return ret;
    }
    /* Prints address in format 0x...*/
    inline std::string to_hex(std::uintptr_t value) {
      std::stringstream ss;
      ss << std::hex << value;
      return ss.str();
    }

  };  // namespace string
};    // namespace utils

#endif  // AXOMAE_UTILS_H
