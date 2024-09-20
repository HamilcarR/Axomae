#ifndef GENERICLOCKINTERFACE_H
#define GENERICLOCKINTERFACE_H
#include <mutex>

/**
 * @file GenericLockInterface.h
 * Provide thread locking mechanisms
 */

/**
 * @class GenericLockInterface
 */
class GenericLockInterface {

 public:
  virtual ~GenericLockInterface() = default;
  virtual void lock() = 0;
  virtual void unlock() = 0;

 protected:
  std::mutex mutex;
};

#endif