#ifndef FACTORY_H
#define FACTORY_H

/*Allows us to bypass private constructor access restriction from std::make_unique construction process*/
template<class TYPE, class... Args>
class PRVINTERFACE : public TYPE {
 public:
  PRVINTERFACE(Args &&...args) : TYPE(std::forward<Args>(args)...) {}
};

#endif