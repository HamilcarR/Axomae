#ifndef PROGRAMOPTIONSMANAGER_H
#define PROGRAMOPTIONSMANAGER_H

/**
 * @brief Command line arguments splitter
 */

namespace controller::cmd {
  class API;
  class ProgramOptionsManager {
   private:
    API *api;

   public:
    /* Takes a valid address of an API object*/
    explicit ProgramOptionsManager(API *api);
    void processArgs(int argv, char **argc);
  };
}  // namespace controller::cmd
#endif