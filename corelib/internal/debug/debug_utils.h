#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#if defined(__GNUC__) || defined(__clang__)
#define ax_dbg_optimize0 __attribute((optimize("O0")))
#define ax_dbg_optimize1 __attribute((optimize("O1")))
#define ax_dbg_optimize2 __attribute((optimize("O2")))
#endif

#endif //DEBUG_UTILS_H
