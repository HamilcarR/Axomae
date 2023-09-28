#include "../includes/Logger.h"
#include <map>

std::map<LogLevel::LOGENUMTYPE , std::string> level_str = {
    {LogLevel::CRITICAL , "CRITICAL"} , 
    {LogLevel::DEBUG , "DEBUG"} , 
    {LogLevel::ERROR , "ERROR"} , 
    {LogLevel::INFO , "INFO"} , 
    {LogLevel::WARNING , "WARNING"},
    {LogLevel::GLINFO , "OPENGL INFO"}
};

#define LOG2STR(level) level_str[level]
#define STR2LOG(str) level_str.find(str)

static std::string getFormatedLog(const std::string& message , const std::string& filename , const unsigned line , LogLevel::LOGENUMTYPE level){
    std::string head = filename + ":" + std::to_string(line) + ";" + LOG2STR(level) + ";" + message; 
    return head ; 
}

AbstractLogger::AbstractLogger(){

}

AbstractLogger::~AbstractLogger(){

}

Logger::Logger(){

}

Logger::~Logger(){

}

void Logger::logMessage(const std::string& message , LogLevel::LOGENUMTYPE log_level , const char* file , unsigned line){
    if(log_level != LogLevel::GLINFO)
        std::cout << getFormatedLog(message , file , line , log_level) << std::endl; 
}

void Logger::logMessage(const char* message , LogLevel::LOGENUMTYPE log_level , const char* file , unsigned line){
    logMessage(std::string(message) , log_level , file , line); 
}

void Logger::print() const {
     
}