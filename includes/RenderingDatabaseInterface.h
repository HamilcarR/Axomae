#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H

#include "Mutex.h"

//TODO: [AX-9] Finish the implementation of the Database interface
template<class T>
class RenderingDatabaseInterface {
public:
    virtual void clean() = 0; 
    virtual void purge() = 0 ;
    virtual T* get(const int id)  = 0; 
    virtual bool contains(const int id) = 0;
protected:
    Mutex mutex; 
};





#endif