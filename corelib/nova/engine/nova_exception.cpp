#include "nova_exception.h"
namespace nova::exception {

  NovaException::NovaException(NovaException &&move) noexcept {
    if (this != &move) {
      auto target_val = move.err_flag.exchange(0, std::memory_order_relaxed);
      err_flag.store(target_val, std::memory_order_relaxed);
    }
  }
  NovaException &NovaException::operator=(NovaException &&move) noexcept {
    if (this != &move) {
      auto target_val = move.err_flag.exchange(0, std::memory_order_relaxed);
      err_flag.store(target_val, std::memory_order_relaxed);
    }
    return *this;
  }
  NovaException::NovaException(const NovaException &copy) noexcept {
    if (this != &copy) {
      auto target_val = copy.err_flag.load(std::memory_order_relaxed);
      err_flag.store(target_val, std::memory_order_relaxed);
    }
  }

  NovaException &NovaException::operator=(const NovaException &copy) noexcept {
    if (this != &copy) {
      auto target_val = copy.err_flag.load(std::memory_order_relaxed);
      err_flag.store(target_val, std::memory_order_relaxed);
    }
    return *this;
  }
  std::vector<ERROR> NovaException::getErrorList() const {
    if (err_flag == NOERR)
      return {NOERR};
    std::vector<ERROR> error_vector;
    error_vector.reserve(64);
    uint64_t padding = 0;
    uint64_t err_type_int = err_flag;
    while (padding < 63) {
      uint64_t test_var = err_type_int & (1ULL << padding);
      if (test_var != 0) {
        error_vector.push_back(static_cast<ERROR>(test_var));
      }
      padding++;
    }
    return error_vector;
  }

  void NovaException::addErrorType(uint64_t to_add) { err_flag.fetch_or(to_add, std::memory_order_relaxed); }

  void NovaException::merge(uint64_t other_error_flag) { err_flag.fetch_or(other_error_flag, std::memory_order_relaxed); }

  // TODO :  #ifdef __CUDA_ARCH__
}  // namespace nova::exception