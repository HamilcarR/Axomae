#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <thread>
template<class T>
class ThreadPool {
 private:
  unsigned int num_threads;
  boost::asio::thread_pool;

 public:
  ThreadPool() { num_threads = std::thread::hardware_concurrency(); }
  void post();
  void join();
};

#endif  // THREADPOOL_H
