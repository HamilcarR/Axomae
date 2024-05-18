#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace threading {

  struct Task {
    std::function<void()> function;
    bool terminable;
  };

  class ThreadPool {
   public:
    int busy_threads;

   private:
    mutable std::mutex mutex;
    std::condition_variable condition_variable;
    std::vector<std::thread> threads;
    bool shutdown_requested{false};
    std::queue<Task> queue;

   public:
    explicit ThreadPool(const int size) : busy_threads(size), threads(std::vector<std::thread>(size)) {
      shutdown_requested = false;
      for (size_t i = 0; i < size; ++i) {
        threads[i] = std::thread(ThreadWorker(this));
      }
    }

    ~ThreadPool() { Shutdown(); }
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    unsigned long threadNumber() const { return threads.size(); }

    void fence() {
      std::unique_lock lock(mutex);
      condition_variable.wait(lock, [this] { return busy_threads == 0; });
    }
    // Waits until threads finish their current task and shutdowns the pool
    void Shutdown() {
      {

        emptyQueue();
        std::lock_guard<std::mutex> lock(mutex);
        shutdown_requested = true;
        condition_variable.notify_all();
      }

      for (auto &thread : threads) {
        if (thread.joinable()) {
          thread.join();
        }
      }
    }

    void emptyQueue() {
      std::lock_guard<std::mutex> lock(mutex);
      std::queue<Task> to_finish;
      while (!queue.empty()) {
        Task elem = queue.front();
        queue.pop();
        if (!elem.terminable)
          to_finish.push(elem);
      }
      queue = to_finish;
    }

    template<typename F, typename... Args>
    auto addTask(bool terminable, F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
      auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
      auto wrapper_func = [task_ptr]() { (*task_ptr)(); };
      Task task{wrapper_func, terminable};
      {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(task);
        // Wake up one thread if its waiting
        condition_variable.notify_one();
      }
      // Return future from promise
      return task_ptr->get_future();
    }

    size_t QueueSize() const {
      std::unique_lock<std::mutex> lock(mutex);
      return queue.size();
    }

   private:
    class ThreadWorker {

     private:
      ThreadPool *thread_pool;

     public:
      explicit ThreadWorker(ThreadPool *pool) : thread_pool(pool) {}

      void operator()() {
        std::unique_lock<std::mutex> lock(thread_pool->mutex);
        while (!thread_pool->shutdown_requested || (thread_pool->shutdown_requested && !thread_pool->queue.empty())) {
          thread_pool->busy_threads--;
          thread_pool->condition_variable.notify_all();
          thread_pool->condition_variable.wait(lock, [this] { return this->thread_pool->shutdown_requested || !this->thread_pool->queue.empty(); });
          thread_pool->busy_threads++;

          if (!this->thread_pool->queue.empty()) {
            auto func = thread_pool->queue.front().function;
            thread_pool->queue.pop();

            lock.unlock();
            func();
            lock.lock();
          }
        }
      }
    };
  };
}  // namespace threading
#endif  // THREADPOOL_H
