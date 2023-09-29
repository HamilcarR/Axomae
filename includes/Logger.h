#ifndef LOGGER_H
#define LOGGER_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>




#define LOG(message , level) Logger::logMessage(message , level , __FILE__ , __LINE__)

namespace LogLevel{
    enum LOGENUMTYPE : unsigned {
        INFO , 
        DEBUG , 
        WARNING , 
        ERROR , 
        CRITICAL, 
        GLINFO
    };
};//End namespace LogLevel


class LogLine{
public:
   LogLine(const std::string& message , const LogLevel::LOGENUMTYPE log_level = LogLevel::INFO); 
   virtual ~LogLine(); 
};

class AbstractLogger{
public:
    AbstractLogger(); 
    virtual ~AbstractLogger();
    virtual void print() const = 0 ; 
};

class Logger : virtual public AbstractLogger{
public:
    Logger(); 
    virtual ~Logger();
    virtual void print() const ;
    static void logMessage(const std::string& message , LogLevel::LOGENUMTYPE log_level , const char* filename , unsigned line);
    static void logMessage(const char* message , LogLevel::LOGENUMTYPE log_level , const char* filename , unsigned line); 
protected:
    std::vector<LogLine> log_buffer ;  
};















#endif