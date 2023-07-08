#ifndef PERFORMANCELOGGER_H
#define PERFORMANCELOGGER_H
#include "Logger.h"

/**
 * @file PerformanceLogger.h
 * This file implements a class that can measure time differences 
 * 
 */

/**
 * @class PerformanceLogger
 * 
 */
class PerformanceLogger : public Logger{
public:
    
    /**
     * @brief Construct a new Performance Logger object
     * 
     */
    PerformanceLogger() : Logger(){
    }

    /**
     * @brief Destroy the Performance Logger object
     * 
     */
    virtual ~PerformanceLogger(){}
    
    /**
     * @brief Store the present timestamp in variable start
     * 
     */
    void startTimer(){ 
        std::cout << "Timer started!\n" ; 
        start = std::chrono::high_resolution_clock::now();  
    } 
    
    /**
     * @brief Store the present timestamp in variable end
     * 
     */
    void endTimer(){
        end = std::chrono::high_resolution_clock::now();  
        std::cout << "Timer end!\n"; 
    }
    
    /**
     * @brief Print the difference between time at end , and time at start 
     * 
     */
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