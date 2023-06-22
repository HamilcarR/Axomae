#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H


//TODO: [AX-9] Finish the implementation of the Database interface
class RenderingDatabaseInterface {
public:
    virtual void softCleanse(); 
    virtual void hardCleanse(); 
    virtual RenderingDatabaseInterface* get(unsigned id); 
    virtual bool contains(unsigned id); 
};





#endif