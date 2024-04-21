#ifndef MUTEX_H
#define MUTEX_H
#include "GenericLockInterface.h"

/**
 * @file Mutex.h
 * This file implements classes for managing resource synchronization , and locks
 *
 */

/**
 * @class Mutex
 * @brief This class uses the RAII principle to lock / unlock the member mutex. So it should be used when a scope lock
 * is needed.
 */
class Mutex : public GenericLockInterface {
 public:
  ~Mutex() override = default;
  class Lock {
   public:
    explicit Lock(Mutex &_mutex);
    virtual ~Lock();
    Lock(const Lock &) = delete;
    Lock(Lock &&) = delete;
    Lock &operator=(const Lock &) = delete;
    Lock &operator=(Lock &&) = delete;

   private:
    Mutex &lock_mutex;
  };

 private:
  void lock() override;
  void unlock() override;
};

#endif