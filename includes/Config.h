#ifndef CONFIG_H
#define CONFIG_H

#include "constants.h"

/**
 * @brief File implementing a utility returning configurations states for the application , from the configuration file. 
 * @file Config.h 
 */

//?see : TODO [AX-27] Create a class reading a config file
class ApplicationConfig{
public:
    ApplicationConfig(){} 
    virtual ~ApplicationConfig(){}
    static std::string getLogFile(){         
        time_t now = time(0); 
        tm *ltm = localtime(&now); 
        auto y = 1900 + ltm->tm_year; 
        auto M = 1 + ltm->tm_mon ; 
        auto d = ltm->tm_mday ; 
        auto h = 5 + ltm->tm_hour ; 
        auto m = 30 + ltm->tm_min ; 
        auto s = ltm->tm_sec ; 
        return std::string("Axomae_log-") + std::to_string(y) + "-" + std::to_string(M) + "-" + std::to_string(d) + "-" + std::to_string(h) + "_" + std::to_string(m) + "_" + std::to_string(s) ; 
    }
private:
};
































#endif