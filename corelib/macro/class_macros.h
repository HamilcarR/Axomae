#ifndef CLASS_MACROS_H
#define CLASS_MACROS_H

// clang-format off

#define GENERATE_GETTERS(return_type, func_field_name, variable) \
  return_type &get## func_field_name() { return variable; } \
  const return_type &get## func_field_name() const { return variable; }

#define GENERATE_SETTERS(variable_type, func_field_name , variable)\
  void set## func_field_name(variable_type var){variable = var;}

#define CLASS_CM(classname) \
  classname() = default;\
  ~classname() = default;\
  classname(const classname& copy) = default;\
  classname(classname&& move) noexcept = default;\
  classname& operator=(const classname& copy) = default;\
  classname& operator=(classname&& move) noexcept = default;

#define CLASS_OCM(classname) \
  classname() = default;\
  ~classname() override = default;\
  classname(const classname& copy) = default;\
  classname(classname&& move) noexcept = default;\
  classname& operator=(const classname& copy) = default;\
  classname& operator=(classname&& move) noexcept = default;

#define CLASS_OM(classname) \
  classname() = default;\
  ~classname() override = default;\
  classname(const classname& copy) = delete;\
  classname(classname&& move) noexcept = default;\
  classname& operator=(const classname& copy) = delete;\
  classname& operator=(classname&& move) noexcept = default;

#define CLASS_C(classname) \
  classname() = default;\
  ~classname() = default;\
  classname(const classname& copy) = default;\
  classname(classname&& move) noexcept = delete;\
  classname& operator=(const classname& copy) = default;\
  classname& operator=(classname&& move) noexcept = delete;

#define CLASS_OC(classname) \
  classname() = default;\
  ~classname() override = default;\
  classname(const classname& copy) = default;\
  classname(classname&& move)noexcept = delete;\
  classname& operator=(const classname& copy) = default;\
  classname& operator=(classname&& move) noexcept = delete;

#define CLASS_O(classname) \
  classname() = default;\
  ~classname() override = default;\
  classname(const classname& copy) = delete;\
  classname(classname&& move) noexcept = delete;\
  classname& operator=(const classname& copy) = delete;\
  classname& operator=(classname&& move) noexcept = delete;

#define CLASS_M(classname) \
  classname() = default;\
  ~classname() = default;\
  classname(const classname& copy) = delete;\
  classname(classname&& move) noexcept = default;\
  classname& operator=(const classname& copy) = delete;\
  classname& operator=(classname&& move) noexcept = default;

#define CLASS_VCM(classname) \
  classname() = default;\
  virtual ~classname() = default;\
  classname(const classname& copy) = default;\
  classname(classname&& move) noexcept = default;\
  classname& operator=(const classname& copy) = default;\
  classname& operator=(classname&& move) noexcept = default;

#define CLASS_VM(classname) \
  classname() = default;\
  virtual ~classname() = default;\
  classname(const classname& copy) = delete;\
  classname(classname&& move) noexcept = default;\
  classname& operator=(const classname& copy) = delete;\
  classname& operator=(classname&& move) noexcept = default;

#define CLASS_VC(classname) \
  classname() = default;\
  virtual ~classname() = default;\
  classname(const classname& copy) = default;\
  classname(classname&& move)noexcept = delete;\
  classname& operator=(const classname& copy) = default;\
  classname& operator=(classname&& move) noexcept = delete;

#define CLASS_V(classname) \
  classname() = default;\
  virtual ~classname() = default;\
  classname(const classname& copy) = delete;\
  classname(classname&& move) noexcept = delete;\
  classname& operator=(const classname& copy) = delete;\
  classname& operator=(classname&& move) noexcept = delete;

// clang-format on
#endif  // CLASS_MACROS_H
