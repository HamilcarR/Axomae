#ifndef AXOMAE_MACROS_H
#define AXOMAE_MACROS_H
#include "Logger.h"

#define ASSERT_SUBTYPE(BASETYPE, SUBTYPE) static_assert(std::is_base_of<BASETYPE, SUBTYPE>::value)
#endif