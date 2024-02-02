
#ifndef OPERATOR_H
#define OPERATOR_H
#include "ILockable.h"
#include "constants.h"

/**
 * @file Operator.h
 * @brief API interface between a GUI element / cmd line option , and the logic interacting with it.
 */
namespace controller {
  namespace ioperator {
    template<class T>
    class OpData {
     public:
      explicit OpData(T data_) : data(data_) {}
      T data;
    };

    template<class MODULE, class DATA_TYPE>
    class OperatorInterface : public ILockable {
     public:
      virtual bool op(ioperator::OpData<DATA_TYPE> *data) const = 0;

     protected:
      MODULE *module;
    };

    template<class MODULE, class DATA_TYPE>
    class UiOperatorInterface : public OperatorInterface<MODULE, DATA_TYPE> {
     public:
      /**
       * @brief Resets element to default state
       */
      virtual void reset() const = 0;
    };
  }  // namespace ioperator
}  // namespace controller
#endif
