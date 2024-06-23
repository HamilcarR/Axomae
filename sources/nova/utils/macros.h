#ifndef MACROS_H
#define MACROS_H
#include "Axomae_macros.h"

/* Macro defining the add_ressource procedure of a Resource Holder*/
#define RESOURCES_DEFINE_ADD(function_name, object_interface, resources_collection) \
  template<class SUBTYPE, class... Args> \
  object_interface *add_##function_name(Args &&...args) { \
    ASSERT_SUBTYPE(object_interface, SUBTYPE); \
    resources_collection.push_back(std::make_unique<SUBTYPE>(std::forward<Args>(args)...)); \
    return resources_collection.back().get(); \
  }

/* Macro defining the create procedure of a resource */
#define RESOURCES_DEFINE_CREATE(object_interface) \
  template<class SUBTYPE, class... Args> \
  static std::unique_ptr<object_interface> create(Args &&...args) { \
    ASSERT_SUBTYPE(object_interface, SUBTYPE); \
    return std::make_unique<SUBTYPE>(std::forward<Args>(args)...); \
  }

#endif  // MACROS_H
