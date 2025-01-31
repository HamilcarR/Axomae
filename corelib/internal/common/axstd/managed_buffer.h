#ifndef MANAGED_BUFFER_H
#define MANAGED_BUFFER_H
#include <cstddef>
#include <cstdlib>
#include <internal/common/utils.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_resource_data.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/macro/project_macros.h>
#include <new>
#include <vector>
namespace axstd {

  class DeviceManagedAllocationPolicy {
   protected:
    void *allocate(std::size_t size_bytes, std::size_t /*align*/) {
      auto result = device::gpgpu::allocate_device_managed(size_bytes, true);
      DEVICE_ERROR_CHECK(result.error_status);
      return result.device_ptr;
    }

    void deallocate(void *ptr, std::size_t /*align*/) {
      auto result = device::gpgpu::deallocate_buffer(ptr);
      DEVICE_ERROR_CHECK(result.error_status);
    }
  };

  class HostManagedAllocationPolicy {
   protected:
    void *allocate(std::size_t size_bytes, std::size_t align) { return ::operator new(size_bytes, std::align_val_t(align)); }

    void deallocate(void *ptr, std::size_t align) { ::operator delete(ptr, std::align_val_t(align)); }
  };

  template<bool is_using_gpu>
  struct AllocationPolicy : protected std::conditional_t<is_using_gpu, DeviceManagedAllocationPolicy, HostManagedAllocationPolicy> {};

  template<class T, bool is_using_gpu = core::build::is_gpu_build, class Policy = AllocationPolicy<is_using_gpu>>
  class UnifiedMemoryAllocator : protected Policy {
   public:
    using value_type = T;
    using reference = T &;
    using pointer = T *;
    using const_reference = const T &;
    using const_pointer = const T *;
    using void_pointer = void *;
    using const_void_pointer = const void *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template<class U>
    struct rebind {
      using other = UnifiedMemoryAllocator<U, is_using_gpu>;
    };

    UnifiedMemoryAllocator() = default;

    T *allocate(size_type size) { return static_cast<T *>(Policy::allocate(size * sizeof(T), alignof(T))); }

    void deallocate(T *ptr, size_type /*size*/) { Policy::deallocate(ptr, alignof(T)); }
  };

  template<class T, class U>
  bool operator==(const UnifiedMemoryAllocator<T> &lhs, const UnifiedMemoryAllocator<U> &rhs) {
    return true;
  }

  template<class T, class U>
  bool operator!=(const UnifiedMemoryAllocator<T> &lhs, const UnifiedMemoryAllocator<U> &rhs) {
    return false;
  }

  template<class T>
  using managed_vector = std::vector<T, UnifiedMemoryAllocator<T>>;
  template<class T>
  using device_managed_vector = std::vector<T, UnifiedMemoryAllocator<T, true>>;
}  // namespace axstd

#endif
