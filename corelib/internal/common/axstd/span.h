#ifndef SPAN_H
#define SPAN_H


#if __cplusplus < 202002L
#include <type_traits>
#include <cstdlib>

namespace axstd{


/* Implementation of a span :
 * The main goal is to provide a usable collection on GPU for C++ versions < 20 and usage in CUDA.
 * Previous implementation using boost is not compatible with CUDA as well.
 */

  template<class T>
  class span {
  public:
    using value_type = typename std::remove_cv<T>::type;
    using element_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using const_pointer = const T*;
    using const_reference = const T&;

  private:
    pointer ptr{};
    size_type len{0};
  public:

    constexpr span() noexcept = default;
    constexpr span(pointer data_, size_type size_) noexcept : ptr(data_), len(size_) {}

    template<class Container>
    constexpr span(Container& container) noexcept : ptr(container.data()), len(container.size()) {
      static_assert(std::is_trivially_copyable_v<typename Container::pointer> , "Container must be trivially copyable");
    }

    template<class Container>
    constexpr span& operator=(Container& container) noexcept {
      static_assert(std::is_trivially_copyable_v<typename Container::pointer> , "Container must be trivially copyable");
      ptr = container.data();
      len = container.size();
      return *this;
    }

    constexpr reference front() const noexcept { return *ptr; }
    constexpr reference back() const noexcept { return *(ptr + len - 1); }
    constexpr reference operator[](size_type index) const noexcept {return ptr[index];}
    constexpr pointer data() const noexcept {return ptr; }
    constexpr size_type size() const noexcept { return len; }
    constexpr size_type size_bytes() const noexcept { return sizeof(T) * len; }
    constexpr bool empty() const noexcept { return len == 0; }

    template<size_type offset , size_type length>
    constexpr span subspan() const noexcept {
      return span(ptr + offset, length);
    }

    constexpr span subspan(size_type offset, size_type length) const noexcept {
      return span(ptr + offset, length);
    }

    template<size_type count>
    constexpr span first() const noexcept {
      return span(ptr, count);
    }

    constexpr span first(size_type count) const noexcept {
      return span(ptr, count);
    }

     template<size_type count>
    constexpr span last() const noexcept {
      return span(ptr + count, len - count);
    }

    constexpr span last(size_type count) const noexcept {
      return span(ptr + count, len - count);
    }


  };
}
#else
  #include <span>
namespace axstd{
template<class T , std::size_t E = std::dynamic_extent>
using span = std::span<T , E>;
}
#endif





#endif //SPAN_H
