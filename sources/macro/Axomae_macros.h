#ifndef AXOMAE_MACROS_H
#define AXOMAE_MACROS_H
#include <cassert>
#include <cstdlib>
#include <type_traits>
#define ISTYPE(TYPE1, TYPE2) std::is_same_v<TYPE1, TYPE2>
#define IS_ARITHMETHIC(TYPE) std::is_arithmetic_v<TYPE>
#define ISSUBTYPE(BASE, DERIVED) std::is_base_of_v<BASE, DERIVED>
#define ASSERT_SUBTYPE(BASETYPE, SUBTYPE) static_assert(ISSUBTYPE(BASETYPE, SUBTYPE))
#define ASSERT_ISTYPE(TYPE1, TYPE2) static_assert(ISTYPE(TYPE1, TYPE2))
#define ASSERT_IS_ARITHMETIC(TYPE) static_assert(IS_ARITHMETHIC(TYPE))
#define NOT_IMPLEMENTED static_assert(false, "Not yet implemented")
// Function does nothing
#define EMPTY_FUNCBODY return;
// clang-format off
#define AX_ASSERT(expr, message) assert(expr && message)
// clang-format on
#define AX_UNREACHABLE assert(false && "Unreachable code executed!");

#define AX_ASSERT_NOTNULL(expr) assert(expr != nullptr)
#define AX_ASSERT_FALSE(expr) assert(!expr)
#define AX_ASSERT_TRUE(expr) assert(expr)
#define AX_ASSERT_NEQ(expr1, expr2) assert(epxr1 != expr2)
#define AX_ASSERT_EQ(expr1, expr2) assert(expr1 == expr2)
#define AX_ASSERT_LE(expr1, expr2) assert(expr1 <= expr2)
#define AX_ASSERT_GE(expr1, expr2) assert(expr1 >= expr2)
#define AX_ASSERT_GT(expr1, expr2) assert(expr1 > expr2)
#define AX_ASSERT_LT(expr1, expr2) assert(expr1 < expr2)

#endif