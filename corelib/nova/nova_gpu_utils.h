#ifndef NOVA_GPU_UTILS_H
#define NOVA_GPU_UTILS_H

#include "internal/common/axstd/span.h"
#include <cstdint>
#include <vector>

/* defines functions and interfaces for nova to communicate with the device */

namespace nova {
  using cache_collection_t = std::vector<axstd::span<uint8_t>>;

  /* Tracks memory pool buffers that are potentially shared with devices.
   * Is cleared before scene loading , and rebuilt after.
   */
  struct device_shared_caches_t {
    /* Takes multiple buffers containing collections of objects... For ex :
     * index 0 : All textures raw data
     * index 1 : All geometry for mesh
     * ...
     */
    cache_collection_t contiguous_caches;

    void addSharedCacheAddress(axstd::span<uint8_t> buffer) { contiguous_caches.push_back(buffer); }
    void clear() { contiguous_caches.clear(); }
  };
}  // namespace nova
namespace nova::gputils {

  void lock_host_memory_default(const device_shared_caches_t &collection);
  void unlock_host_memory(const device_shared_caches_t &collection);

}  // namespace nova::gputils
#endif  // NOVA_GPU_UTILS_H
