#ifndef TAGUTILS_H
#define TAGUTILS_H
#include "device_utils.h"
#include "project_macros.h"
#include "type_list.h"

namespace core::tagutils {

  template<class F, class R, class T0>
  AX_DEVICE_CALLABLE R dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_EQ(tag_index, 0);
    return func(reinterpret_cast<const T0 *>(ptr));
  }

  template<class F, class R, class T0>
  AX_DEVICE_CALLABLE R dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_EQ(tag_index, 0);
    return func(reinterpret_cast<T0 *>(ptr));
  }

  template<class F, class R, class T0>
  R host_dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_EQ(tag_index, 0);
    return func(reinterpret_cast<const T0 *>(ptr));
  }

  template<class F, class R, class T0>
  R host_dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_EQ(tag_index, 0);
    return func(reinterpret_cast<T0 *>(ptr));
  }
  /******************************************************************************/
  template<class F, class R, class T0, class T1>
  AX_DEVICE_CALLABLE R dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 1);
    if (tag_index == 0)
      return func(reinterpret_cast<const T0 *>(ptr));
    return func(reinterpret_cast<const T1 *>(ptr));
  }

  template<class F, class R, class T0, class T1>
  AX_DEVICE_CALLABLE R dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 1);
    if (tag_index == 0)
      return func(reinterpret_cast<T0 *>(ptr));
    return func(reinterpret_cast<T1 *>(ptr));
  }

  template<class F, class R, class T0, class T1>
  R host_dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 1);

    if (tag_index == 0)
      return func(reinterpret_cast<const T0 *>(ptr));
    return func(reinterpret_cast<const T1 *>(ptr));
  }

  template<class F, class R, class T0, class T1>
  R host_dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 1);

    if (tag_index == 0)
      return func(reinterpret_cast<T0 *>(ptr));
    return func(reinterpret_cast<T1 *>(ptr));
  }

  /******************************************************************************/
  template<class F, class R, class T0, class T1, class T2>
  AX_DEVICE_CALLABLE R dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 2);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<const T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<const T1 *>(ptr));
      default:
        return func(reinterpret_cast<const T2 *>(ptr));
    }
  }

  template<class F, class R, class T0, class T1, class T2>
  AX_DEVICE_CALLABLE R dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 2);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<T1 *>(ptr));
      default:
        return func(reinterpret_cast<T2 *>(ptr));
    }
  }

  template<class F, class R, class T0, class T1, class T2>
  R host_dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 2);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<const T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<const T1 *>(ptr));
      default:
        return func(reinterpret_cast<const T2 *>(ptr));
    }
  }

  template<class F, class R, class T0, class T1, class T2>
  R host_dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 2);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<T1 *>(ptr));
      default:
        return func(reinterpret_cast<T2 *>(ptr));
    }
  }
  /******************************************************************************/

  template<class F, class R, class T0, class T1, class T2, class T3>
  AX_DEVICE_CALLABLE R dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 3);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<const T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<const T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<const T2 *>(ptr));
      default:
        return func(reinterpret_cast<const T3 *>(ptr));
    }
  }

  template<class F, class R, class T0, class T1, class T2, class T3>
  AX_DEVICE_CALLABLE R dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 3);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<T2 *>(ptr));
      default:
        return func(reinterpret_cast<T3 *>(ptr));
    }
  }

  template<class F, class R, class T0, class T1, class T2, class T3>
  R host_dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 3);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<const T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<const T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<const T2 *>(ptr));
      default:
        return func(reinterpret_cast<const T3 *>(ptr));
    }
  }

  template<class F, class R, class T0, class T1, class T2, class T3>
  R host_dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    AX_ASSERT_LE(tag_index, 3);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<T2 *>(ptr));
      default:
        return func(reinterpret_cast<T3 *>(ptr));
    }
  }
  /******************************************************************************/

  template<class F, class R, class T0, class T1, class T2, class T3, class... Ts, typename = std::enable_if_t<(sizeof...(Ts) > 0)>>
  AX_DEVICE_CALLABLE R dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<const T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<const T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<const T2 *>(ptr));
      case 3:
        return func(reinterpret_cast<const T3 *>(ptr));
      default:
        return dispatch<F, R, Ts...>(func, ptr, tag_index - 4);
    }
  }
  template<class F, class R, class T0, class T1, class T2, class T3, class... Ts, typename = std::enable_if_t<(sizeof...(Ts) > 0)>>
  AX_DEVICE_CALLABLE R dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<T2 *>(ptr));
      case 3:
        return func(reinterpret_cast<T3 *>(ptr));
      default:
        return dispatch<F, R, Ts...>(func, ptr, tag_index - 4);
    }
  }
  template<class F, class R, class T0, class T1, class T2, class T3, class... Ts, typename = std::enable_if_t<(sizeof...(Ts) > 0)>>
  R host_dispatch(F &&func, void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<T2 *>(ptr));
      case 3:
        return func(reinterpret_cast<T3 *>(ptr));
      default:
        return host_dispatch<F, R, Ts...>(func, ptr, tag_index - 4);
    }
  }

  template<class F, class R, class T0, class T1, class T2, class T3, class... Ts, typename = std::enable_if_t<(sizeof...(Ts) > 0)>>
  R host_dispatch(F &&func, const void *ptr, int tag_index) {
    AX_ASSERT_GE(tag_index, 0);
    switch (tag_index) {
      case 0:
        return func(reinterpret_cast<const T0 *>(ptr));
      case 1:
        return func(reinterpret_cast<const T1 *>(ptr));
      case 2:
        return func(reinterpret_cast<const T2 *>(ptr));
      case 3:
        return func(reinterpret_cast<const T3 *>(ptr));
      default:
        return host_dispatch<F, R, Ts...>(func, ptr, tag_index - 4);
    }
  }

}  // namespace core::tagutils

#endif  // TAGUTILS_H
