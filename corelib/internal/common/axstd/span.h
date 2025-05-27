#ifndef SPAN_H
#define SPAN_H

#include <internal/macro/project_macros.h>
#if __cplusplus < 202002L
#  include <cstdlib>
#  include <internal/device/gpgpu/device_utils.h>
#  include <type_traits>
namespace axstd {

  /*
   * Provides a view on a single region of memory.
   * Can be used on GPU memory.
   */
  template<class T>
  class span {
   public:
    using value_type = typename std::remove_cv<T>::type;
    using element_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;
    using const_pointer = const T *;
    using const_reference = const T &;
    using iterator = pointer;
    using const_iterator = const_pointer;

   private:
    /* Beginning address of the span.*/
    pointer ptr{};
    /* Number of elements in the span.*/
    size_type len{0};

   public:
    ax_device_callable constexpr span() noexcept = default;
    ax_device_callable constexpr span(pointer data_, size_type size_) noexcept : ptr(data_), len(size_) {}

    template<class Container>
    constexpr span(Container &container) noexcept : ptr(container.data()), len(container.size()) {
      static_assert(std::is_trivially_copyable_v<typename Container::pointer>, "Container must be trivially copyable");
    }

    template<class Container>
    constexpr span &operator=(Container &container) noexcept {
      static_assert(std::is_trivially_copyable_v<typename Container::pointer>, "Container must be trivially copyable");
      ptr = container.data();
      len = container.size();
      return *this;
    }

    template<class Container>
    constexpr span(const Container &container) noexcept : ptr(container.data()), len(container.size()) {
      static_assert(std::is_trivially_copyable_v<typename Container::pointer>, "Container must be trivially copyable");
    }

    template<class Container>
    constexpr span &operator=(const Container &container) noexcept {
      static_assert(std::is_trivially_copyable_v<typename Container::pointer>, "Container must be trivially copyable");
      ptr = container.data();
      len = container.size();
      return *this;
    }

    ax_device_callable constexpr reference front() const noexcept { return *ptr; }
    ax_device_callable constexpr reference back() const noexcept { return *(ptr + len - 1); }
    ax_device_callable constexpr reference operator[](size_type index) const noexcept {
      AX_ASSERT_LT(index, len);
      AX_ASSERT_NOTNULL(ptr);
      AX_ASSERT_GT(len, 0);
      return ptr[index];
    }
    ax_device_callable constexpr pointer data() const noexcept { return ptr; }
    ax_device_callable constexpr size_type size() const noexcept { return len; }
    ax_device_callable constexpr size_type size_bytes() const noexcept { return sizeof(T) * len; }
    ax_device_callable constexpr bool empty() const noexcept { return len == 0; }

    template<size_type offset, size_type length>
    ax_device_callable constexpr span subspan() const noexcept {
      return span(ptr + offset, length);
    }

    ax_device_callable constexpr span subspan(size_type offset, size_type length) const noexcept { return span(ptr + offset, length); }

    template<size_type count>
    ax_device_callable constexpr span first() const noexcept {
      return span(ptr, count);
    }

    ax_device_callable constexpr span first(size_type count) const noexcept { return span(ptr, count); }

    template<size_type count>
    ax_device_callable constexpr span last() const noexcept {
      return span(ptr + count, len - count);
    }

    ax_device_callable constexpr span last(size_type count) const noexcept { return span(ptr + count, len - count); }

    ax_device_callable constexpr iterator begin() noexcept { return ptr; }
    ax_device_callable constexpr iterator end() noexcept { return ptr + len; }
    ax_device_callable constexpr const_iterator begin() const noexcept { return ptr; }
    ax_device_callable constexpr const_iterator end() const noexcept { return ptr + len; }
    ax_device_callable constexpr const_iterator cbegin() const noexcept { return ptr; }
    ax_device_callable constexpr const_iterator cend() const noexcept { return ptr + len; }
  };
}  // namespace axstd
#else
#  include <span>
namespace axstd {
  template<class T, std::size_t E = std::dynamic_extent>
  using span = std::span<T, E>;
}
#endif

#endif  // SPAN_H
