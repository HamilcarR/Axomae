#ifndef MEMORYARENA_H
#define MEMORYARENA_H
#include "project_macros.h"
#include <cstdlib>
#include <list>
#include <new>
#include <stdexcept>
#include <thread>
/*
 * Implementation of a simple memory pool.
 * The goal here is to have a system that can allocate a chunk of memory , provide addresses within that chunk to allocate objects with different
 * sizes , and deallocate the entire chunk when not needed.
 */
namespace core::memory {
  template<class T = std::byte>
  class Arena {
   private:
    struct block_t {
      T *current_block_ptr{nullptr};
      std::size_t current_alloc_size{};
    };

    using value_type = T;
    using block_list_t = std::list<block_t>;

   private:
    static constexpr std::size_t L1_DEF_ALIGN = 64;

   private:
    const std::size_t block_size;
    T *current_block_ptr{nullptr};
    std::size_t current_block_offset{}, current_alloc_size{};
    block_list_t used_blocks, free_blocks;

   public:
    Arena(const Arena &) = delete;
    Arena(Arena &&) noexcept = default;
    Arena &operator=(const Arena &) = delete;
    Arena &operator=(Arena &&) noexcept = default;

    explicit Arena(std::size_t block_size_ = 262144) : block_size(block_size_) {}

    ~Arena() {
      freeAlign(current_block_ptr);
      for (auto &block : used_blocks)
        freeAlign(block.current_block_ptr);
      for (auto &block : free_blocks)
        freeAlign(block.current_block_ptr);
    }

    std::size_t getTotalSize() {
      std::size_t acc = current_alloc_size;
      for (const auto &elem : used_blocks)
        acc += elem.current_alloc_size;
      for (const auto &elem : free_blocks)
        acc += elem.current_alloc_size;

      return acc;
    }

    void reset() {
      current_block_offset = 0;
      free_blocks.splice(free_blocks.begin(), free_blocks);
    }

    template<class U>
    U *construct(std::size_t num_instances) {
      U *memory_alloc = static_cast<U *>(alloc(num_instances * sizeof(U)));
      for (std::size_t i = 0; i < num_instances; i++)
        ::new (&memory_alloc[i]) U();
      return memory_alloc;
    }

    void *alloc(std::size_t size_bytes) {
      /* Needs 16 bytes alignment */
      size_bytes = (size_bytes + 0xF) & ~0xF;
      if (current_block_offset + size_bytes > current_alloc_size) {
        /* Adds current block to used list*/
        if (current_block_ptr) {
          used_blocks.push_back({current_block_ptr, current_alloc_size});
          current_block_ptr = nullptr;
        }
        /* Looks for a free block in the free_block list with enough memory*/
        for (typename block_list_t::const_iterator block = free_blocks.begin(); block != free_blocks.end(); block++) {
          if (block->current_alloc_size >= size_bytes) {
            current_block_ptr = block->current_block_ptr;
            current_alloc_size = block->current_alloc_size;
            free_blocks.erase(block);
            break;
          }
        }

        if (!current_block_ptr) {
          current_alloc_size = std::max(size_bytes, block_size);
          current_block_ptr = allocAlign<T>(current_alloc_size);
        }
        current_block_offset = 0;
      }
      T *ret_ptr = current_block_ptr + current_alloc_size;
      current_block_offset += size_bytes;
      return ret_ptr;
    }

   private:
    void *allocAlign(std::size_t size_bytes, std::size_t alignment = L1_DEF_ALIGN) { return aligned_alloc(alignment, size_bytes); }

    template<class U>
    U *allocAlign(std::size_t count, std::size_t alignment = L1_DEF_ALIGN) {
      return static_cast<U *>(allocAlign(count * sizeof(U), alignment));
    }

    void freeAlign(void *ptr) { std::free(ptr); }
  };

}  // namespace core::memory

#endif  // MEMORYARENA_H
