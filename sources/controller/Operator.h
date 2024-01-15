
#ifndef OPERATOR_H
#define OPERATOR_H
#include "constants.h"

/**
 * @file Operator.h
 * @brief API interface between a GUI element , and the logic interacting with it.
 */
namespace controller {

  template<class TYPE>
  class IOperator {
   public:
    virtual bool op() const = 0;

   protected:
    TYPE *widget;
  };
};  // namespace controller
#endif
