#ifndef UTILSMACROS_H
#define UTILSMACROS_H
#include "project_macros.h"
/* Macro defining the create procedure of a resource */
#define RESOURCES_DEFINE_CREATE(object_interface) \
  template<class SUBTYPE, class... Args> \
  static std::unique_ptr<object_interface> create(Args &&...args) { \
    ASSERT_SUBTYPE(object_interface, SUBTYPE); \
    return std::make_unique<SUBTYPE>(std::forward<Args>(args)...); \
  }

/* Macro defining the add_ressource procedure of a Resource Holder*/
#define RESOURCES_DEFINE_ADD(function_name, object_interface, resources_collection) \
  template<class SUBTYPE, class... Args> \
  object_interface *add_##function_name(Args &&...args) { \
    ASSERT_SUBTYPE(object_interface, SUBTYPE); \
    resources_collection.push_back(std::make_unique<SUBTYPE>(std::forward<Args>(args)...)); \
    return resources_collection.back().get(); \
  }
#define RESOURCES_DEFINE_CLEAR(resources_collection) \
  void clear() { resources_collection.clear(); }

#define RESOURCES_DEFINE_GET(function_name, object_interface, resources_collection) \
  std::vector<std::unique_ptr<object_interface>> &get_##function_name() { return resources_collection; }

#define REGISTER_RESOURCE(function_name, object_interface, resources_collection) \
  RESOURCES_DEFINE_ADD(function_name, object_interface, resources_collection) \
  RESOURCES_DEFINE_GET(resources_collection, object_interface, resources_collection) \
  RESOURCES_DEFINE_CLEAR(resources_collection)

#endif  // MACROS_H
