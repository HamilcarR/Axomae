#ifndef CMDARGS_H
#define CMDARGS_H
#include "constants.h"

/**
 * @brief Command line arguments splitter
 */

namespace cmd {
  namespace process {
    struct CommandData {
      std::string command;
      std::vector<std::string> arguments;
    };

    class ProgramArgs {
     public:
      static std::vector<CommandData> processArgs(int argv, char **argc);
    };
  }  // namespace process
}  // namespace cmd
#endif