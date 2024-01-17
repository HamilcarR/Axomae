
#ifndef OPERATOR_H
#define OPERATOR_H
#include "constants.h"

/**
 * @file Operator.h
 * @brief API interface between a GUI element , and the logic interacting with it.
 */
namespace controller {
  namespace ioperator {
    template<class T>
    class OpData {
     public:
      explicit OpData(T data_) : data(data_) {}
      T data;
    };

    template<class WIDGET_TYPE, class DATA_TYPE>
    class IOperator {
     public:
      virtual bool op(ioperator::OpData<DATA_TYPE> *data) const = 0;
      virtual void reset() const = 0;
      virtual ~IOperator() = default;

     protected:
      WIDGET_TYPE *widget;
    };
  }  // namespace ioperator
}  // namespace controller
#endif
