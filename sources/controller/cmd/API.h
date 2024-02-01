#ifndef API_H
#define API_H
#include "Config.h"
#include "Operator.h"
#include "api_namespaces.h"

/**
 * @file API.h
 * @brief This class is responsible for booting either the default state of the program , or setting the program to
 * a specific configuration depending on command line options.
 */

namespace controller::cmd {
  class API {
   public:
    explicit API(int &argv, char **argc);

    const ApplicationConfig &getConfig() { return config; }
    void disableLogging();
    void enableLogging();
    void enableGui();
    void disableGui();
    void bakeTexture(const texturing::INPUTENVMAPDATA &data);
    void enableGpu();
    void configure();
    void configureDefault();
    void viewer(const std::string &file);

   private:
    int *argv;
    char **argc;
    ApplicationConfig config;
  };

}  // namespace controller::cmd

#endif
