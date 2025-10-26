#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cstdint>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace axstd {

  template<size_t SIZE_BYTE, size_t ALIGN = 8>
  class StaticAllocator {
    alignas(ALIGN) uint8_t buffer[SIZE_BYTE];
    size_t offset_bytes = 0;

   public:
    ax_device_callable_inlined static size_t align(size_t value, size_t align) { return (value + align - 1) & ~(align - 1); }

    ax_device_callable_inlined StaticAllocator() = default;

    template<class T, class... Args>
    ax_device_callable_inlined T *construct(Args &&...obj_arg) {
      size_t align_address = align(reinterpret_cast<std::uintptr_t>(buffer) + offset_bytes, alignof(T));
      offset_bytes = align_address - reinterpret_cast<std::uintptr_t>(buffer);
      if (offset_bytes >= SIZE_BYTE)
        return nullptr;
      T *obj = new (&buffer[offset_bytes]) T(std::forward<Args>(obj_arg)...);
      offset_bytes += sizeof(T);
      return obj;
    }

    ax_device_callable_inlined void reset() { offset_bytes = 0; }

    ax_device_callable_inlined bool valid() { return offset_bytes < SIZE_BYTE; }
  };
  using StaticAllocator64kb = StaticAllocator<64 * 1024, 64>;
  using StaticAllocator2kb = StaticAllocator<2048>;
  using StaticAllocator1kb = StaticAllocator<1024>;
  using StaticAllocator512b = StaticAllocator<512>;
  using StaticAllocator256b = StaticAllocator<256>;
  using StaticAllocator128b = StaticAllocator<128>;
  using StaticAllocator64b = StaticAllocator<64>;
  using StaticAllocator32b = StaticAllocator<32>;
  using StaticAllocator16b = StaticAllocator<16>;
}  // namespace axstd

#endif  // MEMORYPOOLALLOCATOR_H
