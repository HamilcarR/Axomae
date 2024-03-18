#ifndef API_H
#define API_H
#include "Config.h"

#include "api_namespaces.h"

/**
 * @file API.h
 * @brief This class is responsible for booting either the default state of the program , or setting the program to
 * a specific configuration depending on command line options.
 */

namespace controller::cmd {
  class API {
   private:
    int *argv;
    char **argc;
    ApplicationConfig config;

   public:
    explicit API(int &argv, char **argc);
    const ApplicationConfig &getConfig() const { return config; }
    void disableLogging();
    void enableLogging();
    void enableEditor();
    void disableEditor();
    void bakeTexture(const texturing::INPUTENVMAPDATA &data);
    void enableGpu();
    void configure();
    void configureDefault();
    void launchHdrTextureViewer(const std::string &file);
    void setUvEditorOptions(const uv::UVEDITORDATA &data);
  };

}  // namespace controller::cmd

#endif
