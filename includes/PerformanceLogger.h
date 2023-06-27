#ifndef PERFORMANCELOGGER_H
#define PERFORMANCELOGGER_H


#include "Logger.h"
//TODO: [AX-23] Implements a chrono giving performance stats



class PerformanceLogger : public Logger{
public:
    PerformanceLogger() : Logger(){

    }
    virtual ~PerformanceLogger(){}
    
    void startTimer(){ 
        std::cout << "Timer started!\n" ; 
        start = std::chrono::high_resolution_clock::now();  
    } 
    
    void endTimer(){
        end = std::chrono::high_resolution_clock::now();  
        std::cout << "Timer end!\n"; 
    }
    
    void print(){
        duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start); 
        std::cout << "Duration : " << duration.count() << "ms\n" ; 
    }
protected:
    std::chrono::high_resolution_clock::time_point start; 
    std::chrono::high_resolution_clock::time_point end ; 
    std::chrono::milliseconds duration ; 




};


#endif