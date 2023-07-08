#ifndef GENERICLOCKINTERFACE_H
#define GENERICLOCKINTERFACE_H
#include <mutex>
#include <cstdlib>

/**
 * @file GenericLockInterface.h
 * Provide thread locking mechanisms
 */

/**
 * @class GenericLockInterface 
 * 
 */
class GenericLockInterface{
public:
    virtual void lock() = 0 ; 
    virtual void unlock() = 0 ; 

protected:
    std::mutex mutex ; 
};











#endif