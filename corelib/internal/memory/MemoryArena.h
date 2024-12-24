#ifndef MEMORYARENA_H
#define MEMORYARENA_H
#include "internal/debug/debug_utils.h"
#include "internal/macro/project_macros.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <list>
#include <new>
#include <unordered_map>

/*
 * Implementation of a simple memory pool.
 * The goal here is to have a system that can allocate a chunk of memory , provide addresses within that chunk to allocate objects with same
 * sizes , and deallocate the entire chunk when not needed.
 * It doesn't support destruction of objects yet. Deallocating blocks will not call the destructor of allocated objects , this is intended as a way to
 * store small struct like objects.
 * Deallocation of individual resource should be made by the caller.This structures doesn't execute destructors.
 */
namespace core::memory {
  constexpr std::size_t B256_ALIGN = 256;
  constexpr std::size_t B128_ALIGN = 128;
  constexpr std::size_t B64_ALIGN = 64;
  constexpr std::size_t B32_ALIGN = 32;
  constexpr std::size_t B16_ALIGN = 16;
  constexpr std::size_t B8_ALIGN = 8;
#if defined(AXOMAE_USE_CUDA)
  constexpr std::size_t PLATFORM_ALIGN = B256_ALIGN;
#else
  constexpr std::size_t PLATFORM_ALIGN = alignof(std::max_align_t);
#endif
  constexpr std::size_t DEFAULT_BLOCK_SIZE = 262144;
  template<class T = std::byte>
  class MemoryArena {
   private:
    using value_type = T;
    /* Keep track of named allocations... useful for debug */
    using tag_map_t = std::unordered_map<const void *, const char *>;
    using ptr_list_t = std::list<void *>;

    struct block_t {
      value_type *current_block_ptr{nullptr};
      std::size_t current_alignment{};
      std::size_t current_alloc_size{};
    };
    struct const_block_t {
      const value_type *current_block_ptr{nullptr};
      std::size_t current_alignment{};
      std::size_t current_alloc_size{};
    };

    /* Keep track of the list of allocated buffers inside a block*/
    using internal_block_alloc_map_t = std::unordered_map<const void *, ptr_list_t>;
    using block_list_t = std::list<block_t>;

   private:
    std::size_t block_size;
    /* This is implicit block, used block list can be empty ,
     * but current_block_ptr can be used , so there's in fact 1 used block even though used_blocks is empty*/
    block_t current_block;
    std::size_t current_block_offset{};
    block_list_t used_blocks, free_blocks;
    tag_map_t tag_map;
    internal_block_alloc_map_t internal_block_alloc_map;

   public:
    MemoryArena(const MemoryArena &) = delete;
    MemoryArena &operator=(const MemoryArena &) = delete;

    MemoryArena(MemoryArena &&) noexcept = default;
    MemoryArena &operator=(MemoryArena &&) noexcept = default;

    explicit MemoryArena(std::size_t block_size_ = DEFAULT_BLOCK_SIZE) : block_size(block_size_) {}

    ~MemoryArena() {
      freeAlign(current_block.current_block_ptr, current_block.current_alignment);
      for (auto &block : used_blocks)
        freeAlign(block.current_block_ptr, block.current_alignment);
      for (auto &block : free_blocks)
        freeAlign(block.current_block_ptr, block.current_alignment);
    }

    ax_no_discard std::size_t getFreeBlocksNum() const { return free_blocks.size(); }

    ax_no_discard std::size_t getUsedBlocksNum() const { return used_blocks.size(); }

    ax_no_discard std::size_t getCurrentAlignment() const { return current_block.current_alignment; }

    const block_list_t &getFreeBlocks() const { return free_blocks; }

    const block_list_t &getUsedBlocks() const { return used_blocks; }

    std::size_t getTotalSize() {
      std::size_t acc = current_block.current_alloc_size;
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
    U *construct(std::size_t num_instances, bool constructor = true, const char *tag = "", std::size_t alignment = PLATFORM_ALIGN) noexcept {
      U *memory_alloc = static_cast<U *>(allocate(num_instances * sizeof(U), tag, alignment));
      if (constructor) {
        for (std::size_t i = 0; i < num_instances; i++)
          ::new (&memory_alloc[i]) U();
      }
      return memory_alloc;
    }

    template<class U, class... Args>
    static U *construct(U *ptr, Args &&...args) noexcept {
      if (!ptr)
        return nullptr;
      ::new (ptr) U(std::forward<Args>(args)...);
      return ptr;
    }

    template<class U, class... Args>
    U *constructAtMemPosition(U *start_buffer_address, std::size_t cache_element_position, Args &&...args) {
      block_t present_block = searchBlock(start_buffer_address);
      if (!present_block.current_block_ptr)
        return nullptr;

      AX_ASSERT_EQ(reinterpret_cast<uintptr_t>(start_buffer_address) % present_block.current_alignment, 0);
      std::size_t offset = cache_element_position * sizeof(U);
      uintptr_t new_address = reinterpret_cast<uintptr_t>(start_buffer_address) + offset;
      return construct(reinterpret_cast<U *>(new_address), std::forward<Args>(args)...);
    }

    /**
     * Copy data to a buffer in a block and returns an aligned pointer to it.
     * @param size : size in bytes
     * @param offset : copy offset from the beginning address of dest , in bytes
     *
     */
    void *copyRange(const void *src, void *dest, std::size_t size, std::size_t offset) noexcept {
      if (!src || !dest)
        return nullptr;
      block_t present_block = searchBlock(dest);
      if (!present_block.current_block_ptr)
        return nullptr;
      if (size > present_block.current_alloc_size)
        return nullptr;

      uintptr_t new_address = reinterpret_cast<uintptr_t>(dest) + offset;
      if (new_address % present_block.current_alignment != 0) {
        new_address = compute_alignment(new_address, present_block.current_alignment);
      }
      std::memcpy(reinterpret_cast<void *>(new_address), src, size);
      return reinterpret_cast<void *>(new_address);
    }

    /* Allocates a buffer of size_bytes size and returns it's desired aligned address. */
    void *allocate(std::size_t size_bytes, const char *tag = "", std::size_t alignment = PLATFORM_ALIGN) {
      size_bytes = compute_alignment(size_bytes, alignment);
      if (current_block_offset + size_bytes > current_block.current_alloc_size) {
        /* Adds current block to used list*/
        addCurrentBlockToUsed();
        /* Looks for a free block in the free_block list with enough memory.*/  // TODO : Replace with best fit rather than first fit.
        for (typename block_list_t::const_iterator block = free_blocks.begin(); block != free_blocks.end(); block++) {
          if (block->current_alloc_size >= size_bytes) {
            current_block = *block;
            free_blocks.erase(block);
            break;
          }
        }

        if (!current_block.current_block_ptr) {
          current_block.current_alloc_size = std::max(size_bytes, block_size);
          current_block.current_block_ptr = allocAlign<T>(current_block.current_alloc_size, alignment);
          current_block.current_alignment = alignment;
        }
        current_block_offset = 0;
      }
      T *ret_ptr = current_block.current_block_ptr + current_block_offset;
      current_block_offset += size_bytes;
      std::memset(ret_ptr, 0x00, size_bytes);
      tag_map[static_cast<const void *>(ret_ptr)] = tag;
      internal_block_alloc_map[static_cast<const void *>(current_block.current_block_ptr)].push_back(ret_ptr);
      return ret_ptr;
    }

    /*
     * deallocate will move the used block to the free list, it will not call delete , only the destructor will properly release memory to the OS.
     */
    void deallocate(void *ptr) {
      if (!ptr)
        return;
      tag_map.erase(ptr);
      if (current_block.current_block_ptr == ptr)
        addCurrentBlockToFree();
      auto to_dealloc = getUsedBlockItr(ptr);
      if (to_dealloc == used_blocks.end())
        return;
      moveUsedToFree(to_dealloc);
    }

   private:
    std::size_t compute_alignment(std::size_t value, std::size_t align = PLATFORM_ALIGN) { return (value + align - 1) & ~(align - 1); }

    bool belong(const void *address, const block_t &block) const {
      auto ptr_address = reinterpret_cast<std::uintptr_t>(address);
      auto block_address = reinterpret_cast<std::uintptr_t>(block.current_block_ptr);
      if (ptr_address >= block_address && ptr_address < block_address + block.current_alloc_size)
        return true;
      return false;
    }

    block_t searchBlock(const void *ptr) const {
      if (belong(ptr, current_block))
        return current_block;
      for (const block_t &block : used_blocks)
        if (belong(ptr, block))
          return block;
      return {};
    }
    const_block_t getCurrentBlock() const {
      const_block_t current;
      current.current_block_ptr = current_block.current_block_ptr;
      current.current_alloc_size = current_block.current_alloc_size;
      current.current_alignment = current_block.current_alignment;
      return current;
    }

    template<class U>
    U *allocAlign(std::size_t count, std::size_t align = PLATFORM_ALIGN) {
      std::size_t total_size = count * sizeof(U);
      auto alignment = static_cast<std::align_val_t>(align);
      void *ptr = ::operator new(total_size, alignment);
      return static_cast<U *>(ptr);
    }

    void freeAlign(void *ptr, std::size_t align = PLATFORM_ALIGN) {
      auto alignment = static_cast<std::align_val_t>(align);
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
      if (current_block.current_block_ptr) {
        used_blocks.push_back({current_block.current_block_ptr, current_block.current_alignment, current_block.current_alloc_size});
        current_block.current_block_ptr = nullptr;
        current_block.current_alloc_size = 0;
        current_block_offset = 0;
      }
    }

    void addCurrentBlockToFree() {
      if (current_block.current_block_ptr) {
        free_blocks.push_back({current_block.current_block_ptr, current_block.current_alignment, current_block.current_alloc_size});
        current_block.current_block_ptr = nullptr;
        current_block.current_alloc_size = 0;
        current_block_offset = 0;
      }
    }

    void moveUsedToFree(typename block_list_t::const_iterator &to_move) {
      if (!to_move->current_block_ptr)
        return;
      free_blocks.push_back({to_move->current_block_ptr, to_move->current_alignment, to_move->current_alloc_size});
      used_blocks.erase(to_move);
    }
  };
  using ByteArena = MemoryArena<std::byte>;
}  // namespace core::memory

#endif  // MEMORYARENA_H
