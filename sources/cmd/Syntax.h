#ifndef SYNTAX_H
#define SYNTAX_H
#include "constants.h"
namespace cmd {
  namespace syntax {

    class Validator {
     public:
      static bool validate(const char *cmd);
    };

  }  // namespace syntax
}  // namespace cmd
#endif
