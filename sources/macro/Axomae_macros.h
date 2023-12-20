#ifndef AXOMAE_MACROS_H
#define AXOMAE_MACROS_H
#include "Logger.h"

#define ISTYPE(TYPE1, TYPE2) std::is_same<TYPE1, TYPE2>::value
#define ISSUBTYPE(BASE, DERIVED) std::is_base_of<BASE, DERIVED>::value
#define ASSERT_SUBTYPE(BASETYPE, SUBTYPE) static_assert(ISSUBTYPE(BASETYPE, SUBTYPE))
#define ASSERT_ISTYPE(TYPE1, TYPE2) static_assert(ISTYPE(TYPE1, TYPE2))
#define NOT_IMPLEMENTED static_assert(false, "Not yet implemented")

#endif