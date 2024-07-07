#ifndef EXCEPTION_MACROS_H
#define EXCEPTION_MACROS_H

#define EXPTN_DEFINE(classname, message) \
  class classname final : public GenericException { \
   public: \
    classname() : GenericException() { saveErrorString(message); } \
    explicit classname(const std::string &message_str) : GenericException() { saveErrorString(message_str); } \
  };

#endif  // EXCEPTION_MACROS_H
