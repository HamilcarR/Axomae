#ifndef FACTORY_H
#define FACTORY_H
#include "constants.h"

/*Allows to bypass private constructor for classes that we want to instantiate with std::make_unique*/
template<class TYPE, class... Args>
class PRVINTERFACE : public TYPE {
 public:
  explicit PRVINTERFACE(Args &&...args) : TYPE(std::forward<Args>(args)...) {}
};

#endif