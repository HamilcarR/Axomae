#ifndef PRIMITIVE_DATASTRUCTURES_H
#define PRIMITIVE_DATASTRUCTURES_H

namespace nova::primitive {
  struct primitive_init_record_s {
    std::size_t total_primitive_count{};
    std::size_t geometric_primitive_count{};
  };
}  // namespace nova::primitive

#endif  // PRIMITIVE_DATASTRUCTURES_H
