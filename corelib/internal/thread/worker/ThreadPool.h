#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace threading {
  const char *const ALL_TASK = "_ALL_";
  struct Task {
    std::function<void()> function;
    std::string context_name;
  };

  class ThreadPool {
    using active_working_threads_t = std::unordered_map<std::thread::id, std::string>;

    int busy_threads;
    mutable std::mutex mutex;  // TODO : wtf ???
    std::condition_variable condition_variable;
    std::vector<std::thread> threads;
    bool shutdown_requested{false};
    std::deque<Task> queue;
    active_working_threads_t current_thread_tasks;

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

    void fence(const std::string &tag) {
      std::unique_lock lock(mutex);
      condition_variable.wait(lock, [this, tag] { return tasksFinished(tag); });
    }

    // Waits until threads finish their current task and shutdowns the pool
    void Shutdown() {
      {
        emptyQueue(ALL_TASK);
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

    void emptyQueue(const std::string &context_name) {
      std::lock_guard<std::mutex> lock(mutex);
      std::deque<Task> to_finish;
      while (!queue.empty()) {
        Task elem = queue.front();
        queue.pop_front();
        if (context_name == ALL_TASK || elem.context_name == context_name)
          to_finish.push_back(elem);
      }
      queue = to_finish;
    }

    template<typename F, typename... Args>
    auto addTask(const std::string &tag_id, F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
      auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
      auto wrapper_func = [task_ptr]() { (*task_ptr)(); };
      Task task{wrapper_func, tag_id};
      {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(task);
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
    bool tasksFinished(const std::string &tag) {
      for (const auto &elem : queue) {
        if (elem.context_name == tag)
          return false;
      }
      /* If queue is empty, we check remaining tasks living IN the threads. */
      for (const auto &elem : current_thread_tasks)
        if (elem.second == tag)
          return false;
      return true;
    }

   private:
    class ThreadWorker {
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
            Task task = thread_pool->queue.front();
            auto func = task.function;
            thread_pool->queue.pop_front();
            thread_pool->current_thread_tasks[std::this_thread::get_id()] = task.context_name;
            lock.unlock();
            func();
            lock.lock();
            thread_pool->current_thread_tasks.erase(std::this_thread::get_id());
          }
        }
      }
    };
  };
}  // namespace threading
#endif  // THREADPOOL_H
