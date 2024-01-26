#ifndef API_NAMESPACES_H
#define API_NAMESPACES_H

namespace controller::texturing {
  struct INPUTENVMAPDATA {

    std::string path_input{};
    std::string path_output{};
    unsigned int width_output{};
    unsigned int height_output{};
    unsigned int samples{};
    std::string baketype;
  };

}  // namespace controller::texturing

#endif  // API_NAMESPACES_H
