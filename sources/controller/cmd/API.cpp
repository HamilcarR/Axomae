#include "API.h"
#include "GenericException.h"
#include "Loader.h"
#include "OfflineCubemapProcessing.h"
#include "TextureViewerWidget.h"
#include "constants.h"

namespace controller::cmd {

  API::API(int &argv_, char **argc_) {
    argv = &argv_;
    argc = argc_;
  }

  void API::disableLogging() { config.loggerSetState(false); }
  void API::enableLogging() { config.loggerSetState(true); }
  void API::enableEditor() { config.setEditorLaunched(true); }
  void API::disableEditor() { config.setEditorLaunched(false); }
  void API::enableGpu() { config.setGpu(true); }

  void API::launchHdrTextureViewer(const std::string &file) {
    IO::Loader loader(nullptr);
    try {
      image::ImageHolder<float> data = loader.loadHdr(file.c_str(), false);
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
      image::ImageHolder<float> data = loader.loadHdr(envmap.path_input.c_str(), false);
      EnvmapProcessing<float> process_texture(data.data, data.metadata.width, data.metadata.height, data.metadata.channels);
      std::unique_ptr<TextureData> texture;
      if (envmap.baketype == "irradiance") {
        texture = process_texture.computeDiffuseIrradiance(envmap.width_output, envmap.height_output, envmap.samples, config.usingGpu());
      }
      image::Metadata metadata;
      metadata.width = texture->width;
      metadata.height = texture->height;
      metadata.channels = texture->nb_components;
      image::ImageHolder<float> hdr(texture->f_data, metadata);
      loader.writeHdr(envmap.path_output.c_str(), hdr);
      QApplication qapp(*argv, argc);
      HdrTextureViewerWidget tex(hdr);
      tex.show();
      qapp.exec();
    } catch (const exception::GenericException &e) {
      std::cerr << e.what();
    }
  }

  void API::setUvEditorOptions(const uv::UVEDITORDATA &data) {
    config.setUvEditorNormalProjectionMode(data.projection_type == "tangent");
    config.setUvEditorResolutionWidth(data.resolution_width);
    config.setUvEditorResolutionHeight(data.resolution_height);
  }

}  // namespace controller::cmd