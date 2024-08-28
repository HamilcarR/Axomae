#include "nova_exception.h"
namespace nova::exception {

#if defined(__CUDA_ARCH__) && defined(AXOMAE_USE_CUDA)
  namespace atomic_nmsp = cuda::std;
#else
  namespace atomic_nmsp = std;
#endif

  NovaException::NovaException(NovaException &&move) noexcept {
    if (this != &move) {
      auto target_val = move.err_flag.exchange(0, atomic_nmsp::memory_order_relaxed);
      err_flag.store(target_val, atomic_nmsp::memory_order_relaxed);
    }
  }
  NovaException &NovaException::operator=(NovaException &&move) noexcept {
    if (this != &move) {
      auto target_val = move.err_flag.exchange(0, atomic_nmsp::memory_order_relaxed);
      err_flag.store(target_val, atomic_nmsp::memory_order_relaxed);
    }
    return *this;
  }
  NovaException::NovaException(const NovaException &copy) noexcept {
    if (this != &copy) {
      auto target_val = copy.err_flag.load(atomic_nmsp::memory_order_relaxed);
      err_flag.store(target_val, atomic_nmsp::memory_order_relaxed);
    }
  }

  NovaException &NovaException::operator=(const NovaException &copy) noexcept {
    if (this != &copy) {
      auto target_val = copy.err_flag.load(atomic_nmsp::memory_order_relaxed);
      err_flag.store(target_val, atomic_nmsp::memory_order_relaxed);
    }
    return *this;
  }

  void NovaException::addErrorType(uint64_t to_add) { err_flag.fetch_or(to_add, atomic_nmsp::memory_order_relaxed); }

  void NovaException::merge(uint64_t other_error_flag) { err_flag.fetch_or(other_error_flag, atomic_nmsp::memory_order_relaxed); }

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

}  // namespace nova::exception