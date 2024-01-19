#include "CmdArgs.h"
#include "Syntax.h"
#include "TerminalOpt.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

static constexpr const char *cmds[] = {
    "-q"  // quiet mode
};

namespace cmd {
  namespace syntax {
    bool Validator::validate(const char *cmd) {}
  }  // namespace syntax

  /*****************************************************************************************************************************************************************/

  namespace process {
    std::vector<CommandData> ProgramArgs::processArgs(int argv, char **argc) {
      std::vector<CommandData> commands;
      for (int i = 0; i < argv; i++) {
        if (syntax::Validator::validate(argc[i])) {
          ;
        }
      }
    }
  }  // namespace process
}  // namespace cmd