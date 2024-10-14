#ifndef MEMORYPOOLALLOCATOR_H
#define MEMORYPOOLALLOCATOR_H
#include "MemoryArena.h"
#include <internal/common/exception/GenericException.h>
#include <limits>
#include <memory>

/**
 * Implementation of a stateful heap custom allocator.
 * This allocator uses a MemoryArena as the underlying memory.
 */

namespace core::memory {
  namespace exception {
    class ArenaInvalidStateException : public ::exception::GenericException {
     public:
      ArenaInvalidStateException() : GenericException() { saveErrorString("Invalid memory arena state."); }
    };
  }  // namespace exception

  template<class T>
  class MemoryPoolAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

   private:
    MemoryArena<> *arena{nullptr};
    inline void throwOnInvalidArenaState(const MemoryArena<> *arena) {
      if (!arena)
        throw exception::ArenaInvalidStateException();
    }

   public:
    CLASS_CM(MemoryPoolAllocator)

    explicit MemoryPoolAllocator(MemoryArena<> *arena) : arena{arena} { throwOnInvalidArenaState(arena); }

    template<class U>
    explicit MemoryPoolAllocator(MemoryPoolAllocator<U> &other) noexcept : arena{other.arena} {}

    /**
     * Allocates n elements each of size T
     */
    T *allocate(std::size_t n) {
      throwOnInvalidArenaState(arena);
      if (std::numeric_limits<std::size_t>::max() / sizeof(T) < n)
        throw std::bad_alloc();
      return static_cast<T *>(arena->allocate(n * sizeof(T)));
    }

    void deallocate(void *ptr) {
      throwOnInvalidArenaState(arena);
      arena->deallocate(ptr);
    }
  };

}  // namespace core::memory

#endif  // MEMORYPOOLALLOCATOR_H
