#ifndef API_NAMESPACES_H
#define API_NAMESPACES_H

namespace controller {
  namespace texturing {
    struct INPUTENVMAPDATA {

      std::string path_input{};
      std::string path_output{};
      unsigned int width_output{};
      unsigned int height_output{};
      unsigned int samples{};
      std::string baketype;
    };

  }  // namespace texturing

  namespace uv {
    struct UVEDITORDATA {
      std::string projection_type;
      unsigned int resolution_width;
      unsigned int resolution_height;
    };
  }  // namespace uv

}  // namespace controller
#endif  // API_NAMESPACES_H
