#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H
#include "Mutex.h"


/**
 * @file RenderingDatabaseInterface.h
 * This file implements an interface for databases of objects , like , for example , light databases.
 * 
 */

/**
 * @brief This class provides an interface of pure abstract methods to manage rendering objects databases
 * @class RenderingDatabaseInterface
 * @tparam T Class type of the object stored in the database
 */
template<class U , class T>
class RenderingDatabaseInterface {
public:

    /**
     * @brief Proceeds with a soft clean of the database . The implementation depends on the class that inherits this , but this usually consists of only some objects being freed
     * 
     */
    virtual void clean() = 0;
    
    /**
     * @brief Proceeds with a complete purge of the database . Everything is freed . 
     * 
     */
    virtual void purge() = 0 ;
    
    /**
     * @brief Return a pointer on a single element , after a search using an ID . 
     * @param id ID of the element
     * @return T* Pointer on the element 
     */
    virtual T* get(const U id)  = 0; 

    /**
     * @brief Adds an element in the database
     * 
     * @param element Object to store 
     * @param keep Keep the element between scene change
     */
    virtual U add(T *element , bool keep) = 0 ; 

    /**
     * @brief Checks if database contains an object with specific ID . 
     * 
     * @param id ID of the element sought. 
     */
    virtual bool contains(const U id) = 0;

    /**
     * @brief Checks if the element is present , if element not present , returns nullptr
     * 
     * @param element_address address of the element we check
     * @return T* Pointer on the element  
     */
    virtual std::pair<U , T*> contains(const T* element_address) = 0; 
protected:
    Mutex mutex;        /*<Mutex used for thread safety*/ 
};





#endif