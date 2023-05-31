#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H



class RenderingDatabaseInterface {
public:
    virtual void softCleanse(); 
    virtual void hardCleanse(); 
    virtual RenderingDatabaseInterface* get(unsigned id); 
    virtual bool contains(unsigned id); 
};





#endif