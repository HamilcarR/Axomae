#ifndef SPAN_H
#define SPAN_H

#if __cplusplus < 202002L
#include "boost/core/span.hpp"
namespace axstd{
  template<class T , std::size_t E = boost::dynamic_extent>
  using span = boost::span<T , E>;
  }
#else
  #include <span>
namespace axstd{
template<class T , std::size_t E = std::dynamic_extent>
using span = std::span<T , E>;
}
#endif





#endif //SPAN_H
