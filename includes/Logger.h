#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <cstdlib>
#include <chrono>

class Logger{
public:
    Logger(){}
    virtual ~Logger(){}
    virtual void print() = 0 ; 

};


#endif