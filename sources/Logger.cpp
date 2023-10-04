#include "../includes/Logger.h"
#include "../includes/Mutex.h"
#include "../includes/GenericException.h"
#include <map>

#define LOG2STR(level) level_str[level]
#define STR2LOG(str) level_str.find(str)


static Logger logger_global; 


class LoggerOutputStreamException : virtual public AxomaeGenericException {
public:
    LoggerOutputStreamException() : AxomaeGenericException() {
        saveErrorString("The output stream for the application logger system is undefined !");
    }
    virtual ~LoggerOutputStreamException(){}
};



std::map<LogLevel::LOGENUMTYPE , std::string> level_str = {
    {LogLevel::CRITICAL , "CRITICAL"} , 
    {LogLevel::DEBUG , "DEBUG"} , 
    {LogLevel::ERROR , "ERROR"} , 
    {LogLevel::INFO , "INFO"} , 
    {LogLevel::WARNING , "WARNING"},
    {LogLevel::GLINFO , "OPENGL INFO"}
};


void LogFunctions::log_message(std::string message , const LogLevel::LOGENUMTYPE level , const char* file , const char* function,  unsigned int line){
    logger_global.logMessage(message , level , file , function , line); 
}

void LogFunctions::log_message(const char* message , const LogLevel::LOGENUMTYPE level , const char* file , const char* function , unsigned int line){
    log_message(std::string(message) , level , file , function , line); 
}

void LogFunctions::log_flush(){
    logger_global.flush(); 
}

void LogFunctions::log_configure(const LoggerConfigDataStruct &conf){
    logger_global.setLogSystemConfig(conf); 
}

static std::string getFormatedLog(const std::string& message , const std::string& filename , const std::string& function , const unsigned line , LogLevel::LOGENUMTYPE level){
    std::string head = filename + ": At function: " + function + "() : At Line : " + std::to_string(line) + " ; " + LOG2STR(level) + " ; " + message; 
    return head ; 
}

AbstractLogger::AbstractLogger(){

}

AbstractLogger::~AbstractLogger(){

}

Logger::Logger() : AbstractLogger(){
}

Logger::~Logger(){

}

LogLine::LogLine(const std::string& mes , const LogLevel::LOGENUMTYPE lev , const char* _file , const char* _function , const unsigned _line){
    message = mes ; 
    level = lev ;
    file = _file ; 
    line = _line ;
    function = _function ;

}

LogLine::~LogLine(){

}

std::string LogLine::getFormattedLog() const {
    return getFormatedLog(message , file , function , line , level); 
}

void Logger::logMessage(const std::string& message , LogLevel::LOGENUMTYPE log_level , const char* file , const char* function, unsigned line){
    Mutex::Lock lock(mutex); 
    LogLine elem(message , log_level , file , function, line);
    log_buffer.push_back(elem);
    print(); 
}

void Logger::logMessage(const char* message , LogLevel::LOGENUMTYPE log_level , const char* file , const char* function ,  unsigned line){
    logMessage(std::string(message) , log_level , file , function , line); 
}

void Logger::print() const {
    if(out){
        auto last_log = log_buffer.back();
        *out << last_log.getFormattedLog() << "\n";  
    }
}

void Logger::flush() {
    if(!out)
        throw LoggerOutputStreamException(); 
    if(!stdout_logging){
        Mutex::Lock lock(mutex); 
        for(LogLine elem : log_buffer)
            *out << getFormatedLog(elem.getMessage() , elem.getLoggedFile() , elem.getFunctionName() , elem.getLine() , elem.getLogLevel()) << "\n";   
    }
}

void Logger::setLogSystemConfig(const LoggerConfigDataStruct &config){
    Mutex::Lock lock(mutex); 
    filters = config.log_filters ;
    priority = config.log_level ; 
    out = config.write_destination; 
}