#include "../includes/Mutex.h"



Mutex::Lock::Lock(Mutex& mutex): lock_mutex(mutex){
    lock_mutex.lock();
}

Mutex::Lock::~Lock(){
    lock_mutex.unlock();
}

void Mutex::lock(){
    mutex.lock(); 
}

void Mutex::unlock(){
    mutex.unlock();
}