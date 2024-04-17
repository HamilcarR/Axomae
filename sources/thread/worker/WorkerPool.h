#ifndef WORKERPOOL_H
#define WORKERPOOL_H
#include "ILockable.h"

#include <condition_variable>
#include <cstdlib>
#include <queue>
#include <variant>

template<class T>
class WorkerQueue : protected ILockable {
 public:
  void push(const T &val);
  T pop();

 private:
  std::queue<T> task_queue;
  std::condition_variable cond_var;
};

template<class T>
void WorkerQueue<T>::push(const T &val) {
  Mutex::Lock lock(mutex);
  task_queue.push(val);
  cond_var.notify_one();
}

template<class T>
T WorkerQueue<T>::pop() {
  Mutex::Lock lock(mutex);
  cond_var.wait(mutex, [&] { return !task_queue.empty(); });
  T ret = task_queue.front();
  task_queue.pop();
  return ret;
}

#endif
