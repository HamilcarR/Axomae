#include "API.h"
#include "Loader.h"
#include "TextureProcessing.h"
#include "TextureViewerWidget.h"
#include "constants.h"
#include "internal/common/exception/GenericException.h"

namespace controller::cmd {

  API::API(int &argv_, char **argc_) {
    argv = &argv_;
    argc = argc_;
  }

  void API::disableLogging() { config.flag &= ~CONF_ENABLE_LOGS; }

  void API::initializeThreadPool(int n) {
    config.flag |= CONF_USE_MTHREAD;
    config.initializeThreadPool(n);
  }

  void API::enableLogging() { config.flag |= CONF_ENABLE_LOGS; }

  void API::enableEditor() { config.flag |= CONF_USE_EDITOR; }

  void API::disableEditor() { config.flag &= ~CONF_USE_EDITOR; }

  void API::enableGpu() { config.flag |= CONF_USE_CUDA; }

  void API::launchHdrTextureViewer(const std::string &file) {
    IO::Loader loader(nullptr);
    try {
      auto data = loader.loadHdrEnvmap(file.c_str(), false);
      QApplication qapp(*argv, argc);
      HdrTextureViewerWidget tex(data);
      tex.show();
      qapp.exec();
    } catch (const exception::GenericException &e) {
      std::cerr << e.what();
    }
  }

  void API::bakeTexture(const texturing::INPUTENVMAPDATA &envmap) {
    IO::Loader loader(nullptr);
    try {
      image::ImageHolder<float> data = loader.loadHdrEnvmap(envmap.path_input.c_str(), false);
      TextureOperations<float> process_texture(data.data, data.metadata.width, data.metadata.height, data.metadata.channels);
      image::ImageHolder<float> texture;
      if (envmap.baketype == "irradiance") {
        texture = process_texture.computeDiffuseIrradiance(envmap.width_output, envmap.height_output, envmap.samples, (config.flag & CONF_USE_CUDA));
      }
      loader.writeHdr(envmap.path_output.c_str(), texture);
      QApplication qapp(*argv, argc);
      HdrTextureViewerWidget tex(texture);
      tex.show();
      qapp.exec();
    } catch (const exception::GenericException &e) {
      std::cerr << e.what();
    }
  }

  void API::setUvEditorOptions(const uv::UVEDITORDATA &data) {
    if (data.projection_type == "tangent")
      config.flag |= CONF_UV_TSPACE;
    else if (data.projection_type == "object")
      config.flag |= CONF_UV_OSPACE;
    else {
      std::cerr << "Wrong projection type.";
      config.flag |= CONF_UV_TSPACE;
      return;
    }
    config.setUvEditorResolutionWidth(data.resolution_width);
    config.setUvEditorResolutionHeight(data.resolution_height);
  }

}  // namespace controller::cmd