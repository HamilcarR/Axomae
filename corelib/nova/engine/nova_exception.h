#ifndef EXCEPTION_H
#define EXCEPTION_H
#include <atomic>
#include <class_macros.h>
#include <cstdint>
#include <vector>
namespace nova::exception {

  enum ERROR : uint64_t {
    NOERR = 0,
    INVALID_INTEGRATOR = 1 << 0,
    INVALID_RENDER_MODE = 1 << 1,

    INVALID_SAMPLER_DIM = 1 << 2,
    SAMPLER_INIT_ERROR = 1 << 3,
    SAMPLER_DOMAIN_EXHAUSTED = 1 << 4,  // Internal exception of third parties low discrep generators
    SAMPLER_INVALID_ALLOC = 1 << 5,
    SAMPLER_INVALID_ARG = 1 << 6,

    GENERAL_ERROR = 1ULL << 32,

  };

  class NovaException {
   private:
    std::atomic<uint64_t> err_flag{NOERR};

   public:
    NovaException() = default;
    NovaException(NovaException &&move) noexcept;
    NovaException &operator=(NovaException &&move) noexcept;
    NovaException(const NovaException &copy) noexcept;
    NovaException &operator=(const NovaException &copy) noexcept;
    ~NovaException() = default;

    [[nodiscard]] bool errorCheck() const { return err_flag != NOERR; }
    [[nodiscard]] std::vector<ERROR> getErrorList() const;
    [[nodiscard]] uint64_t getErrorFlag() const { return err_flag; }
    void addErrorType(uint64_t to_add);
    /* merges err_flag and other_error_flag , err_flag will now store it's previous errors + other_error_flag */
    void merge(uint64_t other_error_flag);
  };

}  // namespace nova::exception

#endif  // EXCEPTION_H
