#ifndef ALIASES_H
#define ALIASES_H

#include <internal/memory/Allocator.h>

#ifdef __CUDA_ARCH__
using StackAllocator = axstd::StaticAllocator<2048, 4>;
#else
using StackAllocator = axstd::StaticAllocator64kb;
#endif
#endif