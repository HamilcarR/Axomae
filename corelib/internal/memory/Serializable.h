#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace core::memory {

  class Serializable {
   public:
    virtual ~Serializable() = default;
    virtual std::vector<uint8_t> serialize() const = 0;
  };

}  // namespace core::memory

#endif  // SERIALIZERINTERFACE_H
