#ifndef CMDARGS_H
#define CMDARGS_H
#include "API.h"
#include "constants.h"
/**
 * @brief Command line arguments splitter
 */

namespace controller::cmd {

  class ProgramOptionsManager {
   public:
    explicit ProgramOptionsManager(API &api);
    bool processArgs(int argv, char **argc);

   private:
    API &api;
  };

}  // namespace controller::cmd
#endif