#ifndef TPTR_H
#define TPTR_H
#include "tagutils.h"
#include <cstdint>
#include <utility>
/*
 * This is an implementation of a tag pointer hack. Obviously it is not very portable to other archs than ADM64 , and Aarch64 ,
 * but other cases would be out of scope to this project. (NB : Distributed rendering using smartphone/smart-fridge/arduino/microwave/toaster could be
 * interesting ?  )
 * Needed because :
 * 1) virtual dynamic dispatch between host and device is not really compatible due to the vtable.
 * 2) the padding added by the vtable quickly leaves a big memory footprint when using lot of geometry on Nova.
 * 3) std::variants are not supported on cuda.
 *
 * Considering pointers have an 8 bytes alignment , the last 3 lower bits of a pointer address are 0 .
 * This gives us at the very minimum , 2^3 different possible types to work with.
 * In addition to this , virtual addresses take a canonical form with at least the 7 highest bits unused by the OS.
 * This scales to 2^10 types storable
 *
 * tag mask should be 0xFE 00 00 00 00 00 00 07
 * We will use 2^7 in a first time though for simplicity.
 *
 */

namespace core {
  template<class... Ts>
  class tag_ptr {
   public:
    using type_pack = type_list<Ts...>;

   private:
    static constexpr int shift = 57;
    static constexpr int tagbits = 64 - shift;
    static constexpr uintptr_t tagmask = ~((1ull << shift) - 1);
    static constexpr uintptr_t ptrmask = ~tagmask;
    uintptr_t bits = 0;

   public:
    template<class T>
    ax_device_callable tag_ptr(T *ptr) {
      auto conv = reinterpret_cast<uintptr_t>(ptr);
      constexpr uintptr_t type_index = index<T>();
      bits = (type_index << shift) | conv;
    }

    ax_device_callable tag_ptr(std::nullptr_t) {}

    ax_device_callable tag_ptr() = default;

    ax_device_callable ~tag_ptr() = default;

    ax_device_callable tag_ptr(const tag_ptr &tptr) = default;

    ax_device_callable tag_ptr(tag_ptr &&tptr) noexcept = default;

    ax_device_callable tag_ptr &operator=(const tag_ptr &tptr) = default;

    ax_device_callable tag_ptr &operator=(tag_ptr &&tptr) noexcept = default;

    template<class T>
    ax_device_callable static constexpr unsigned int index() {
      using FT = typename std::remove_cv_t<T>;
      if constexpr (ISTYPE(FT, std::nullptr_t))
        return 0;
      return 1 + type_id<FT, type_pack>::index;
    }

    template<class T>
    ax_device_callable ax_no_discard bool isType() const {
      return tag() == index<T>();
    }

    template<class T>
    ax_device_callable ax_no_discard T *get() {
      if (isType<T>())
        return reinterpret_cast<T *>((bits & ptrmask));
      return nullptr;
    }

    template<class T>
    ax_device_callable ax_no_discard const T *get() const {
      if (isType<T>())
        return reinterpret_cast<const T *>((bits & ptrmask));
      return nullptr;
    }

    template<class F>
    ax_device_callable decltype(auto) dispatch(F &&func) {
      AX_ASSERT_NOTNULL(get());
      using R = typename return_type<F, Ts...>::RETURN_TYPE;
      return tagutils::dispatch<F, R, Ts...>(std::forward<F>(func), get(), tag() - 1);
    }

    template<class F>
    ax_device_callable decltype(auto) dispatch(F &&func) const {
      AX_ASSERT_NOTNULL(get());
      using R = typename return_type<F, Ts...>::RETURN_TYPE;
      return tagutils::dispatch<F, R, Ts...>(std::forward<F>(func), get(), tag() - 1);
    }

    template<class F>
    ax_host_only decltype(auto) host_dispatch(F &&func) {
      AX_ASSERT_NOTNULL(get());
      using R = typename return_type<F, Ts...>::RETURN_TYPE;
      return tagutils::host_dispatch<F, R, Ts...>(std::forward<F>(func), get(), tag() - 1);
    }

    template<class F>
    ax_host_only decltype(auto) host_dispatch(F &&func) const {
      AX_ASSERT_NOTNULL(get());
      using R = typename return_const_type<F, Ts...>::RETURN_TYPE;
      return tagutils::host_dispatch<F, R, Ts...>(std::forward<F>(func), get(), tag() - 1);
    }

    ax_device_callable ax_no_discard unsigned int tag() const { return bits >> shift; }

    ax_device_callable void *get() { return reinterpret_cast<void *>(bits & ptrmask); }

    ax_device_callable ax_no_discard const void *get() const { return reinterpret_cast<const void *>(bits & ptrmask); }

    ax_device_callable static constexpr int tagSize() { return sizeof...(Ts); }

    ax_device_callable static constexpr uintptr_t ptrMask() { return ptrmask; }

    ax_device_callable static constexpr uintptr_t tagMask() { return tagmask; }
  };
}  // namespace core
#endif  // TPTR_H
