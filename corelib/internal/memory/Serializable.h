#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace axstd {

  class Serializable {
   public:
    virtual ~Serializable() = default;
    virtual std::vector<uint8_t> serialize() const = 0;
  };

}  // namespace axstd

#endif  // SERIALIZERINTERFACE_H
