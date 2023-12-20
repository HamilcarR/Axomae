#include "TerminalOpt.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace axomae {

  const int STRING_SIZE = 256;

  template<typename T>
  struct Validation {
    bool validated;
    std::vector<T> command_arguments;
  };
  std::mutex mutex_window_thread1, mutex_window_thread2;
  static const char *CMD_ERROR = "Wrong command used !";
  static const char *LOADERR = "Image loading failed :";
  ProgramStatus *ProgramStatus::instance = nullptr;
  static const std::regex command_regex[] = {
      std::regex("window [0-9]+ [0-9]+ [a-z]+", std::regex_constants::icase),  // open a new SDL window
      std::regex("nmap [0-9]+ () () (-gpu|-cpu)",
                 std::regex_constants::icase),  // generate normal map (from height map)
      std::regex("hmap [0-9]+ (-sobel|-prewitt|-scharr) (-repeat) (-gpu|-cpu)",
                 std::regex_constants::icase),                                          // generate height map (from albedo texture)
      std::regex("dudv [0-9]+ () (-repeat) (-gpu|-cpu)", std::regex_constants::icase),  // generate dudv map
      std::regex("save [0-9]+ (/?([a-zA-Z0-9]+)/?)*[a-zA-Z0-9]+.[a-zA-Z0-9]+",
                 std::regex_constants::icase),                                                         // save image on the disk
      std::regex("contrast [0-9]+ [0-9]+ [a-z]+ (-gpu|-cpu)", std::regex_constants::icase),            // set contrast
      std::regex("exit", std::regex_constants::icase),                                                 // exit the app
      std::regex("render [0-9]+ [0-9]+ [a-z]+", std::regex_constants::icase),                          // render the texture on a mesh
      std::regex("load (/?([a-zA-Z0-9]+)/?)*[a-zA-Z0-9]+.[a-zA-Z0-9]+", std::regex_constants::icase),  // load an image
      std::regex("ls", std::regex_constants::icase),                                                   // list all image ids
      std::regex("select [0-9]+", std::regex_constants::icase),                                        // select image id as work image
      std::regex("id", std::regex_constants::icase)                                                    // check current image id
  };

  /*******************************************************************************************************************************************************/
  static bool check_if_number(std::string &input) {
    bool it = std::all_of(input.begin(), input.end(), [](char c) { return std::isdigit(c); });
    return it;
  }

  /*******************************************************************************************************************************************************/

  /*retrieve an argument from a command*/
  static std::string get_word(std::string &input, Uint8 pos) {
    if (pos > input.size() || input.size() == 0)
      return std::string();
    else {
      char input_c[STRING_SIZE] = " ";
      strcpy(input_c, input.c_str());
      char *tokens = strtok(input_c, " \n");
      int number_args = 0;
      while (tokens) {
        if (pos == number_args)
          return std::string(tokens);
        tokens = strtok(NULL, " \n");
        number_args++;
      }
      return std::string();
    }
  }

  /*******************************************************************************************************************************************************/

  static Validation<std::string> validate_command_load(std::string input) {
    std::string delimiter = " ";
    if (std::regex_match(input, command_regex[LOAD])) {
      std::vector<std::string> arg_array;
      std::string arg1 = get_word(input, 1);
      if (arg1.size() > 0) {
        arg_array.push_back(arg1);
        return {true, arg_array};
      } else
        return {false, std::vector<std::string>()};
    } else
      return {false, std::vector<std::string>()};
  }

  /*******************************************************************************************************************************************************/
  static Validation<std::string> validate_command_save(std::string input) {

    if (std::regex_match(input, command_regex[SAVE])) {
      std::vector<std::string> arg_array;
      std::string arg1 = get_word(input, 1);
      std::string arg2 = get_word(input, 2);
      arg_array.push_back(arg1);
      arg_array.push_back(arg2);
      return {true, arg_array};

    } else
      return {false, std::vector<std::string>()};
  }

  /*******************************************************************************************************************************************************/
  static Validation<std::string> validate_command_window(std::string input) {
    std::string delimiter = " ";
    if (std::regex_match(input, command_regex[WIN])) {
      bool validate_input = false;
      std::vector<std::string> arg_array;
      std::string arg1 = get_word(input, 1);
      std::string arg2 = get_word(input, 2);
      std::string arg3 = get_word(input, 3);
      bool c1 = check_if_number(arg1);
      bool c2 = check_if_number(arg2);
      if (c1 && c2 && arg1.size() > 0 && arg2.size() > 0 && arg3.size() > 0) {
        arg_array.push_back(arg1);
        arg_array.push_back(arg2);
        arg_array.push_back(arg3);
        return {true, arg_array};

      } else
        return {false, std::vector<std::string>()};

    } else
      return {false, std::vector<std::string>()};
  }

  /*******************************************************************************************************************************************************/
  static Validation<std::string> validate_command_contrast(std::string input) { return {false, std::vector<std::string>()}; }

  /*******************************************************************************************************************************************************/

  static Validation<std::string> validate_command_dudv(std::string input) {
    std::string delimiter = " ";

    return {false, std::vector<std::string>()};
  }

  /*******************************************************************************************************************************************************/
  static Validation<std::string> validate_command_nmap(std::string input) { return {false, std::vector<std::string>()}; }

  /*******************************************************************************************************************************************************/

  static Validation<std::string> validate_command_hmap(std::string input) {
    std::string delimiter = " ";
    std::string w1 = get_word(input, 1);
    std::string w2 = get_word(input, 2);
    std::string w3 = get_word(input, 3);
    std::string w4 = get_word(input, 4);
    std::vector<std::string> a;
    a.push_back(w1);
    a.push_back(w2);
    a.push_back(w3);
    a.push_back(w4);
    if ((w2.compare("-prewitt") != 0 && w2.compare("-sobel") != 0 && w2.compare("-scharr") != 0) || !check_if_number(w1) ||
        w3.compare("-repeat") != 0)
      return {false, std::vector<std::string>()};
    return {true, a};
  }

  /*******************************************************************************************************************************************************/
  static Validation<std::string> validate_command_render(std::string input) { return {false, std::vector<std::string>()}; }

  /*******************************************************************************************************************************************************/

  static Validation<std::string> validate_command_select(std::string input) {
    std::string w1 = get_word(input, 1);

    if (std::regex_match(input, command_regex[SELECT])) {
      if (check_if_number(w1)) {
        std::vector<std::string> A;
        A.push_back(w1);
        return {true, A};
      } else
        return {false, std::vector<std::string>()};

    }

    else
      return {false, std::vector<std::string>()};
  }

  /*******************************************************************************************************************************************************/

  /*******************************************************************************************************************************************************/
  static void loop_thread(int width, int height, std::string name) {
    ProgramStatus *instance = ProgramStatus::getInstance();
    Window *display = instance->getDisplay();
    auto images = instance->getImages();
    bool loop = instance->getLoop();
    mutex_window_thread1.lock();
    display = new Window(width, height, name.c_str());
    instance->setDisplay(display);
    mutex_window_thread1.unlock();
    SDL_Event event;
    display->setEvent(event);
    int prev_image_id = instance->getCurrentImageId();
    int image = images.size() - 1;
    while (loop) {
      int _idCurrentImage = (instance->getCurrentImageId() == prev_image_id) ? prev_image_id : instance->getCurrentImageId();
      mutex_window_thread2.lock();
      if (instance->getDisplay() != nullptr && _idCurrentImage >= 0 && (unsigned int)_idCurrentImage < images.size())
        instance->getDisplay()->display_image(images[_idCurrentImage].first);
      mutex_window_thread2.unlock();
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
          loop = false;
      }
      if (!instance->getLoop())
        loop = false;
    }
    mutex_window_thread2.lock();
    display->cleanUp();
    instance->setDisplayNULL();
    mutex_window_thread2.unlock();
  }

  /*******************************************************************************************************************************************************/

  ProgramStatus::ProgramStatus() {
    _idCurrentImage = -1;
    exited = false;
    _display_window = nullptr;
    loop = true;
  }

  void ProgramStatus::setLoop(bool l) {
    mutex_window_thread2.lock();
    loop = l;
    mutex_window_thread2.unlock();
  }

  bool ProgramStatus::getLoop() { return loop; }
  ProgramStatus::~ProgramStatus() { loop = false; }

  void ProgramStatus::exit() { delete instance; }

  void ProgramStatus::create_window(int width, int height, const char *name) {
    if (_display_window == nullptr) {
      std::string n(name);
      void (*ref)(int, int, std::string) = &loop_thread;
      std::thread thread = std::thread(ref, width, height, n);
      _threads.push_back(std::move(thread));
    } else {
      print(std::string("A texture window is already active..."), RED, PROMPT2);
    }
  }

  void ProgramStatus::process_command(std::string user_input) {

    /*does it match ?*/

    bool save = std::regex_search(user_input, command_regex[SAVE]);
    bool load = std::regex_search(user_input, command_regex[LOAD]);
    bool window = std::regex_search(user_input, command_regex[WIN]);
    bool normalmap = std::regex_search(user_input, command_regex[NMAP]);
    bool heightmap = std::regex_search(user_input, command_regex[HMAP]);
    bool contrast = std::regex_search(user_input, command_regex[CONTRAST]);
    bool render = std::regex_search(user_input, command_regex[RENDER]);
    bool dudv = std::regex_search(user_input, command_regex[DUDV]);
    bool closew = std::regex_search(user_input, std::regex("exwin"));
    bool list_ids = std::regex_search(user_input, command_regex[LISTIDS]);
    bool selectid = std::regex_search(user_input, command_regex[SELECT]);
    bool what_selected = std::regex_search(user_input, command_regex[CHK_CURRENT_IMG]);

    if (save) {
      Validation<std::string> v = validate_command_save(user_input);
      int id = atoi(v.command_arguments[0].c_str());
      if (v.validated) {
        if (id >= 0 && (unsigned int)id < images.size()) {
          print(std::string("Saving..."), GREEN, PROMPT2);
          ImageImporter::save_image(images[id].first, v.command_arguments[1].c_str());
          print(std::string("Done."), GREEN, PROMPT2);

        } else {
          print(std::string("Wrong ID image used!"), RED);
        }
      } else {
        print(CMD_ERROR, RED);
      }
    } else if (load) {
      Validation<std::string> v = validate_command_load(user_input);
      if (v.validated) {
        std::string a = std::string("File : " + v.command_arguments[0] + " loading...");
        print(a, GREEN);
        ImageImporter *instance = ImageImporter::getInstance();
        SDL_Surface *im = instance->load_image(static_cast<const char *>(v.command_arguments[0].c_str()));
        if (im)
          images.push_back(std::pair<SDL_Surface *, std::string>(im, v.command_arguments[0]));
        else
          print("Loading image failed !", RED);

      } else {
        print(CMD_ERROR, RED);
      }

    } else if (window) {
      Validation<std::string> v = validate_command_window(user_input);
      if (v.validated) {
        int w = std::stoi(v.command_arguments[0]), h = std::stoi(v.command_arguments[1]);
        std::string window_name = v.command_arguments[2];

        create_window(w, h, window_name.c_str());

      } else
        print(CMD_ERROR, RED);
    } else if (normalmap) {

    } else if (heightmap) {
      Validation<std::string> v = validate_command_hmap(user_input);
      if (v.validated) {
        int id = std::stoi(v.command_arguments[0]);
        std::string func = v.command_arguments[1];
        std::string bord = v.command_arguments[2];
        std::string device_choice = v.command_arguments[3];
        uint8_t f = func.compare("-prewitt") == 0 ? AXOMAE_USE_PREWITT :
                    func.compare("-sobel") == 0   ? AXOMAE_USE_SOBEL :
                    func.compare("-scharr") == 0  ? AXOMAE_USE_SCHARR :
                                                    0;
        uint8_t b = AXOMAE_REPEAT;
        if (device_choice.compare("-gpu") == 0)
          ImageManager::USE_GPU_COMPUTING();
        else
          ImageManager::USE_CPU_COMPUTING();
        ImageManager::set_greyscale_luminance(images[id].first);  // TODO change to hmap
      } else
        print(CMD_ERROR, RED);
    } else if (contrast) {

    } else if (render) {

    } else if (dudv) {

    } else if (list_ids) {
      if (images.size() == 0)
        print(std::string("No image loaded..."), RED);
      else {
        std::string liste = "";
        unsigned int count = 0;
        for (std::pair<SDL_Surface *, std::string> p : images) {
          liste += "-" + std::to_string(count) + " : " + p.second + "\n";
          count++;
        }
        print(liste, BLUE, PROMPT2);
      }
    } else if (closew)
      std::puts("Exiting...\n");
    else if (selectid) {
      Validation<std::string> v = validate_command_select(user_input);
      if (v.validated) {
        try {
          int id = std::stoi(v.command_arguments[0].c_str());
          if ((unsigned int)id >= images.size() || id < 0)
            print(std::string("Invalid id input"), RED);

          else {
            _idCurrentImage = id;
            print(std::string("Selected : " + images[id].second), GREEN, PROMPT2);
          }
        } catch (std::invalid_argument &e) {
          print(std::string("Invalid argument input"), RED);
        }
      }
    } else if (what_selected) {
      std::string a = "";
      if (_idCurrentImage != -1) {
        a = std::to_string(_idCurrentImage);
        a += ": " + images[_idCurrentImage].second;
      } else
        a = "No image ID selected";
      print(a, YELLOW, PROMPT2);
    } else
      print(CMD_ERROR, RED);
  }

  /*******************************************************************************************************************************************************/

  /*******************************************************************************************************************************************************/

}  // namespace axomae
