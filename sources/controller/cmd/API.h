#ifndef API_H
#define API_H
#include "Config.h"

#include "api_namespaces.h"

#include <internal/macro/project_macros.h>

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
    API(int &argv, char **argc, std::thread::id main_thread_id);

    /**
     * Use this method after completing the ApplicationConfig structure states.
     * Will move the final config property , and invalidate the current instance.
     */
    ax_no_discard ApplicationConfig &&getConfig() {
      argv = nullptr;
      argc = nullptr;
      return std::move(config);
    }
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
    void initializeThreadPool(int n);
  };

}  // namespace controller::cmd

#endif
