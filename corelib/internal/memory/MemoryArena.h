#ifndef MEMORYARENA_H
#define MEMORYARENA_H
#include "internal/macro/project_macros.h"
#include <cstdlib>
#include <list>
#include <new>
/*
 * Implementation of a simple memory pool.
 * The goal here is to have a system that can allocate a chunk of memory , provide addresses within that chunk to allocate objects with same
 * sizes , and deallocate the entire chunk when not needed.
 * It doesn't support destruction of objects yet. Deallocating blocks will not call the destructor of allocated objects , this is intended as a way to
 * store small struct like objects.
 * Deallocation of individual resource should be made by the caller.This structures doesn't execute destructors.
 */
namespace core::memory {

  static constexpr std::size_t L1_DEF_ALIGN = 64;
  static constexpr std::size_t DEFAULT_BLOCK_SIZE = 262144;
  template<class T = std::byte>
  class MemoryArena {
   private:
    struct block_t {
      T *current_block_ptr{nullptr};
      std::size_t current_alloc_size{};
    };

    using value_type = T;
    using block_list_t = std::list<block_t>;

   private:
    std::size_t block_size;
    /* This is implicit block, used block list can be empty , but current_block_ptr can be used , so there's in fact 1 used block even though
     * used_blocks is empty*/
    T *current_block_ptr{nullptr};
    std::size_t current_block_offset{}, current_alloc_size{};
    block_list_t used_blocks, free_blocks;

   public:
    MemoryArena(const MemoryArena &) = delete;
    MemoryArena &operator=(const MemoryArena &) = delete;

    MemoryArena(MemoryArena &&) noexcept = default;
    MemoryArena &operator=(MemoryArena &&) noexcept = default;

    MemoryArena(std::size_t block_size_ = DEFAULT_BLOCK_SIZE) : block_size(block_size_) {}

    ~MemoryArena() {
      freeAlign(current_block_ptr);
      for (auto &block : used_blocks)
        freeAlign(block.current_block_ptr);
      for (auto &block : free_blocks)
        freeAlign(block.current_block_ptr);
    }

    std::size_t getFreeBlocksNum() const { return free_blocks.size(); }

    std::size_t getUsedBlocksNum() const { return used_blocks.size(); }

    const block_list_t &getFreeBlocks() const { return free_blocks; }

    const block_list_t &getUsedBlocks() const { return used_blocks; }

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
      free_blocks.splice(free_blocks.begin(), used_blocks);
    }

    template<class U>
    U *construct(std::size_t num_instances, bool constructor = true) {
      U *memory_alloc = static_cast<U *>(allocate(num_instances * sizeof(U)));
      if (constructor) {
        for (std::size_t i = 0; i < num_instances; i++)
          ::new (&memory_alloc[i]) U();
      }
      return memory_alloc;
    }

    template<class U, class... Args>
    static U *construct(U *ptr, Args &&...args) {
      if (!ptr)
        throw std::bad_alloc();
      ::new (ptr) U(std::forward<Args>(args)...);
      return ptr;
    }

    void *allocate(std::size_t size_bytes) {
      /* Needs 16 bytes alignment */
      size_bytes = (size_bytes + 0xF) & ~0xF;
      if (current_block_offset + size_bytes > current_alloc_size) {
        /* Adds current block to used list*/
        addCurrentBlockToUsed();
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
      T *ret_ptr = current_block_ptr + current_block_offset;
      current_block_offset += size_bytes;
      return ret_ptr;
    }

    /*
     * deallocate will move the used block to the free list, it will not call delete , only the destructor will properly release memory to the OS.
     */
    void deallocate(void *ptr) {
      if (!ptr)
        return;
      if (current_block_ptr == ptr)
        addCurrentBlockToFree();
      auto to_dealloc = getUsedBlockItr(ptr);
      if (to_dealloc == used_blocks.end())
        return;
      moveUsedToFree(to_dealloc);
    }

   private:
    template<class U>
    U *allocAlign(std::size_t count) {
      std::size_t total_size = count * sizeof(U);
      auto alignment = static_cast<std::align_val_t>(L1_DEF_ALIGN);
      void *ptr = ::operator new(total_size, alignment);
      return static_cast<U *>(ptr);
    }

    void freeAlign(void *ptr) {
      auto alignment = static_cast<std::align_val_t>(L1_DEF_ALIGN);
      ::operator delete(ptr, alignment);
    }

    typename block_list_t::const_iterator getUsedBlockItr(void *ptr) const {
      T *block_ptr = static_cast<T *>(ptr);
      for (typename block_list_t::const_iterator block = used_blocks.begin(); block != used_blocks.end(); block++) {
        if (block->current_block_ptr == block_ptr)
          return block;
      }
      return used_blocks.end();
    }

    void addCurrentBlockToUsed() {
      if (current_block_ptr) {
        used_blocks.push_back({current_block_ptr, current_alloc_size});
        current_block_ptr = nullptr;
        current_alloc_size = 0;
        current_block_offset = 0;
      }
    }

    void addCurrentBlockToFree() {
      if (current_block_ptr) {
        free_blocks.push_back({current_block_ptr, current_alloc_size});
        current_block_ptr = nullptr;
        current_alloc_size = 0;
        current_block_offset = 0;
      }
    }

    void moveUsedToFree(typename block_list_t::const_iterator &to_move) {
      if (!to_move->current_block_ptr)
        return;
      free_blocks.push_back({to_move->current_block_ptr, to_move->current_alloc_size});
      used_blocks.erase(to_move);
    }
  };
  using ByteArena = MemoryArena<std::byte>;
}  // namespace core::memory

#endif  // MEMORYARENA_H
