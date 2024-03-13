#include "CmdArgs.h"
#include "GenericException.h"
#include "Operator.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <regex>

namespace exception {
  class StrToNumConversionException : public CatastrophicFailureException {
   public:
    explicit StrToNumConversionException(const std::string &wrong_value) : CatastrophicFailureException() {
      CatastrophicFailureException::saveErrorString(std::string("Cannot convert ") + wrong_value + std::string(" to a numerical value"));
    }
  };

  void checkStoiInvalidConversion(const std::string &to_check) {
    try {
      int num_value = std::stoi(to_check);
    } catch (const std::invalid_argument &e) {
      throw StrToNumConversionException(to_check);
    }
  }

}  // namespace exception

static const char *required_command_str = "";
namespace controller::cmd {
  namespace po = boost::program_options;

  ProgramOptionsManager::ProgramOptionsManager(API &api_) : api(api_) {}

  /* Program options*/
  static void configOpts(po::variables_map &vm, API &api, bool &command_valid) {
    api.disableLogging();
    if (vm.count("verbose")) {
      command_valid = true;
      api.enableLogging();
    }
    if (vm.count("gpu")) {
      api.enableGpu();
    }
    if (vm.count("uv")) {
      api.enableEditor();
      command_valid = true;
      const auto &opts = vm["uv"].as<std::vector<std::string>>();
      try {
        exception::checkStoiInvalidConversion(opts[1]);
        exception::checkStoiInvalidConversion(opts[2]);
      } catch (const exception::StrToNumConversionException &e) {
        throw e;
      }
      uv::UVEDITORDATA data;
      data.projection_type = std::string(opts[0]);
      data.resolution_width = std::stoi(opts[1]);
      data.resolution_height = std::stoi(opts[2]);
      api.setUvEditorOptions(data);
    }
  }

  void setDescript(po::options_description &descript) {
    // clang-format off
    descript.add_options()
        ("help,h", "Prints this help message")
        ("verbose,v", "Turn on stdout logs")
        ("editor,e", "Launch the editor")
        ("gpu,g" , "Enable GPGPU compute")
        ("viewer" , po::value<std::string>(), "Open viewer for the specified file")
        ("uv" , po::value<std::vector<std::string>>()->multitoken() ,
        "Usage: \n\
        --uv [tangent|object] [X resolution] [Y resolution]\n\
        Launch the program with the uv editor set to these options")
        ("bake,b",po::value<std::vector<std::string>>()->multitoken(),
        "Usage: \n\
        --bake [type] [width] [height] [samples] [path_in] [path_out]\n\
        Generates a texture of type [type] (irradiance|prefilter)");
    // clang-format on
  }

  void ProgramOptionsManager::processArgs(int argv, char **argc) {
    po::options_description descript("Options");

    /*Sets up program options*/
    setDescript(descript);
    po::variables_map vm;
    po::store(po::parse_command_line(argv, argc, descript), vm);
    po::notify(vm);
    bool valid_command = false; /* if set to true , the program can be launched with only the provided option */
    if (vm.count("help")) {
      std::cout << descript << "\n";
      valid_command = true;
    }
    /* Order is important here , first the cases that only modify the ApplicationConfig states*/
    try {
      configOpts(vm, api, valid_command);
    } catch (const exception::GenericException &e) {
      std::cerr << e.what();
    }
    /* Configure the application */
    api.configure();

    /* Cases that launch a process(task in the future)*/
    if (vm.count("editor")) {
      api.enableEditor();
      valid_command = true;
    }
    if (vm.count("viewer")) {
      valid_command = true;
      api.disableEditor();
      std::string file = std::string(vm["viewer"].as<std::string>());
      api.launchHdrTextureViewer(file);
    }
    if (vm.count("bake")) {
      valid_command = true;
      api.disableEditor();
      const auto &opts = vm["bake"].as<std::vector<std::string>>();
      if (opts.size() != 6) {
        throw po::required_option("Option \"--bake\" missing argument");
      }
      texturing::INPUTENVMAPDATA envmap;
      try {
        for (int i = 1; i <= 3; i++)
          exception::checkStoiInvalidConversion(opts[i]);
      } catch (const exception::StrToNumConversionException &e) {
        throw e;
      }
      envmap.baketype = std::string(opts[0]);
      envmap.width_output = std::stoi(opts[1]);
      envmap.height_output = std::stoi(opts[2]);
      envmap.samples = std::stoi(opts[3]);
      envmap.path_input = std::string(opts[4]);
      envmap.path_output = std::string(opts[5]);
      api.bakeTexture(envmap);  // replace by tasking
    }

    if (!valid_command) {
      throw po::required_option(required_command_str);
    }
  }

}  // namespace controller::cmd