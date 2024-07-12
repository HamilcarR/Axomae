#ifndef EXCEPTION_MACROS_H
#define EXCEPTION_MACROS_H

#define GENERIC_EXCEPT_DEFINE(classname, message, severity) \
  class classname final : public GenericException { \
   public: \
    classname() : GenericException(message, severity) {} \
  };

#define CRITICAL_EXCEPT_DEFINE(classname, message, severity) \
  class classname final : public CatastrophicFailureException { \
   public: \
    classname() : CatastrophicFailureException(message, severity) {}

#endif  // EXCEPTION_MACROS_H
