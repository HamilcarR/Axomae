#ifndef GENERICEXCEPTION_H
#define GENERICEXCEPTION_H
#include <stdexcept>

class AxomaeGenericException : virtual public std::exception{
public:

    AxomaeGenericException() : std::exception(){}
    virtual ~AxomaeGenericException(){} 


    virtual const char* what() const noexcept {
        std::string ret = std::string("The program has encountered an exception:\n") + this_error_string; 
        return ret.c_str() ; 
    }
protected:
    std::string this_error_string ; 
};

















#endif