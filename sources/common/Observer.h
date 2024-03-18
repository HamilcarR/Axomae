#ifndef OBSERVER_H
#define OBSERVER_H
#include "constants.h"

namespace observer {

  /**
   * @brief Serves as a storage generic datatype for passing messages between publishers and subscribers.
   * @note the pointer "data" has no guarantee of staying valid .
   * @tparam T Type of the data
   */
  template<class T>
  class Data {
   public:
    T data;
  };
};  // namespace observer

template<class DATATYPE>
class ISubscriber {
 public:
  virtual bool operator==(const ISubscriber &compare) const { return this == &compare; }
  virtual bool operator!=(const ISubscriber &compare) const { return !(this->operator==(compare)); }
  virtual void notified(observer::Data<DATATYPE> &data) = 0;
};

template<class DATATYPE>
class IPublisher {
 protected:
  std::vector<ISubscriber<DATATYPE> *> subscribers;

 public:
  virtual void notify(observer::Data<DATATYPE> &data) const = 0;
  virtual void attach(ISubscriber<DATATYPE> &subscriber) = 0;
  virtual void detach(ISubscriber<DATATYPE> &subscriber) = 0;
};

#endif