#include "CmdArgs.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace controller::cmd {
  namespace po = boost::program_options;

  ProgramOptionsManager::ProgramOptionsManager(API &api_) : api(api_) {}

  static void configOpts(po::variables_map &vm, API &api) {
    api.disableLogging();
    if (vm.count("verbose")) {
      api.enableLogging();
    }
    if (vm.count("gpu")) {
      api.enableGpu();
    }
    if (vm.count("gui")) {
      api.enableGui();
    }
  }

  void setDescript(po::options_description &descript) {
    // clang-format off
    descript.add_options()
        ("help,h", "Prints this help message")
        ("verbose,v", "Turn on stdout logs")
        ("gui,g", "Launch the editor")
        ("gpu" , "Enable GPGPU compute")
        ("bake,b",po::value<std::vector<std::string>>()->multitoken(),
        "Usage: \n\
        --bake [type] [width] [height] [samples] [path_in] [path_out]\n\
        Generates a texture of type [type] (irradiance|prefilter)");
    // clang-format on
  }

  bool ProgramOptionsManager::processArgs(int argv, char **argc) {
    po::options_description descript("Options");

    /*Sets up program options*/
    setDescript(descript);
    po::variables_map vm;
    po::store(po::parse_command_line(argv, argc, descript), vm);
    po::notify(vm);
    if (vm.count("help")) {
      std::cout << descript << "\n";
    }

    /* Order is important here , first the cases that only modify the ApplicationConfig*/
    configOpts(vm, api);
    api.configure();
    /* Cases that launch a process(task in the future)*/
    if (vm.count("bake")) {
      const auto &opts = vm["bake"].as<std::vector<std::string>>();
      if (opts.size() != 6) {
        throw po::required_option("Option \"--bake\" missing argument");
      }
      texturing::INPUTENVMAPDATA envmap;
      envmap.baketype = std::string(opts[0]);
      envmap.width_output = std::stoi(opts[1]);
      envmap.height_output = std::stoi(opts[2]);
      envmap.samples = std::stoi(opts[3]);
      envmap.path_input = std::string(opts[4]);
      envmap.path_output = std::string(opts[5]);
      api.bakeTexture(envmap);  // replace by tasking
    }
    return true;
  }

}  // namespace controller::cmd