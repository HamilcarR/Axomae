#ifndef MANAGED_BUFFER_H
#define MANAGED_BUFFER_H
#include <cstdlib>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/macro/project_macros.h>

namespace axstd {

  template<class Policy>
  class PolicyAllocation {
   public:
    template<class T>
    T *allocate(std::size_t elements) {
      return static_cast<Policy>(this)->allocate(elements);
    }
    template<class T>
    void deallocate(T *ptr) {
      static_cast<Policy>(this)->deallocate(ptr);
    }
  };

  class SystemAllocationPolicy : public PolicyAllocation<SystemAllocationPolicy> {
   public:
    template<class T>
    T *allocate(std::size_t num_elements) {
      return new T[num_elements];
    }
    template<class T>
    void deallocate(T *ptr) {
      delete[] ptr;
    }
  };

  class DeviceAllocationPolicy : public PolicyAllocation<DeviceAllocationPolicy> {
   public:
    template<class T>
    T *allocate(std::size_t num_elements) {
      device::gpgpu::GPU_query_result result = device::gpgpu::allocate_device_managed(num_elements * sizeof(T), true);
      DEVICE_ERROR_CHECK(result.error_status);
      return static_cast<T *>(result.device_ptr);
    }

    template<class T>
    void deallocate(T *ptr) {
      device::gpgpu::GPU_query_result result = device::gpgpu::deallocate_buffer(ptr);
      DEVICE_ERROR_CHECK(result.error_status);
    }
  };

  /* Wrapper around managed memory . Initialized host side */
  template<class T, class Policy>
  class managed_buffer : protected PolicyAllocation<Policy> {
    using policy = PolicyAllocation<Policy>;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = std::size_t;

    std::size_t size{};
    T *data{};

   public:
    managed_buffer() = default;
    ~managed_buffer() = default;
    managed_buffer(const managed_buffer &copy) = delete;
    managed_buffer &operator=(const managed_buffer &copy) = delete;
    managed_buffer(managed_buffer &&other) noexcept {
      size = std::move(other.size);
      data = std::move(other.data);
      other.size = 0;
      other.data = nullptr;
    }
    managed_buffer &operator=(managed_buffer &&other) noexcept {
      if (this == &other)
        return *this;
      size = std::move(other.size);
      data = std::move(other.data);
      other.size = 0;
      other.data = nullptr;
      return *this;
    }
    void release() {
      policy::deallocate(data);
      size = 0;
      data = nullptr;
    }
    void init(std::size_t num_elements) {
      if (data)
        release();
      size = num_elements;
      data = policy::allocate(num_elements);
    }

    reference operator[](size_type index) {
      AX_ASSERT_LT(index, size);
      return data[index];
    }

    const_reference operator[](size_type index) const {
      AX_ASSERT_LT(index, size);
      return data[index];
    }
  };

}  // namespace axstd

#endif
