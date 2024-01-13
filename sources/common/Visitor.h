#ifndef VISITOR_H
#define VISITOR_H
#include "constants.h"

template<class TYPE>
class IVisitor {
 public:
  virtual bool process(const TYPE *element) const = 0;
};

#endif
