#ifndef AXOMAE_MACROS_H
#define AXOMAE_MACROS_H
#include "constants.h"

#define ISTYPE(TYPE1, TYPE2) std::is_same_v<TYPE1, TYPE2>
#define IS_ARITHMETHIC(TYPE) std::is_arithmetic_v<TYPE>
#define ISSUBTYPE(BASE, DERIVED) std::is_base_of_v<BASE, DERIVED>
#define ASSERT_SUBTYPE(BASETYPE, SUBTYPE) static_assert(ISSUBTYPE(BASETYPE, SUBTYPE))
#define ASSERT_ISTYPE(TYPE1, TYPE2) static_assert(ISTYPE(TYPE1, TYPE2))
#define ASSERT_IS_ARITHMETIC(TYPE) static_assert(IS_ARITHMETHIC(TYPE))
#define NOT_IMPLEMENTED static_assert(false, "Not yet implemented")
// Function does nothing
#define EMPTY_FUNCBODY return
#define AX_ASSERT(expr) assert(expr)
#endif