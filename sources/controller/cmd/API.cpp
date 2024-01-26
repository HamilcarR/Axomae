#include "API.h"
#include "GenericException.h"
#include "Loader.h"
#include "OfflineCubemapProcessing.h"
#include "constants.h"
namespace controller::cmd {

  API::API() {}

  void API::disableLogging() { config.loggerSetState(false); }
  void API::enableLogging() { config.loggerSetState(true); }
  void API::enableGui() { config.setGuiLaunched(true); }
  void API::disableGui() { config.setGuiLaunched(false); }
  void API::enableGpu() { config.setGpu(true); }
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
    } catch (const GenericException &e) {
      std::cerr << e.what();
    }
  }
}  // namespace controller::cmd