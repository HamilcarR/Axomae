#ifndef OBSERVER_H
#define OBSERVER_H
#include "constants.h"

namespace observer {
  template<class T>
  struct Data {
    T data;
  };
};  // namespace observer

template<class DATATYPE>
class ISubscriber {
 public:
  virtual bool operator==(const ISubscriber &compare) const = 0;
  virtual bool operator!=(const ISubscriber &compare) const { return !(this->operator==(compare)); }
  virtual void notified(observer::Data<DATATYPE> data) = 0;
};

template<class DATATYPE>
class IPublisher {
 public:
  virtual void notify(observer::Data<DATATYPE> data) const = 0;
  virtual void attach(ISubscriber<DATATYPE> &subscriber) = 0;
  virtual void detach(ISubscriber<DATATYPE> &subscriber) = 0;
};

#endif