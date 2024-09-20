#ifndef DEBUGGL_H
#define DEBUGGL_H
#include "Logger.h"
#include "gl_headers.h"
#include <iostream>
#define GL_ERROR_CHECK(function) \
  do { \
    function; \
    errorCheck(__FILE__, __func__, __LINE__); \
  } while (0)

inline void errorCheck(const char *filename, const char *function, unsigned int line) {
#ifndef NDEBUG
  GLenum error = GL_NO_ERROR;
  error = glGetError();  // put in lopp
  if (error != GL_NO_ERROR) {
    std::string err = std::string("GL API ERROR :") + std::to_string(error);
    LOGFL(err, LogLevel::DEBUG, filename, function, line);
  }
#endif
}

inline void errorCheck(const char *filename, unsigned int line) {
#ifndef NDEBUG
  GLenum error = GL_NO_ERROR;
  error = glGetError();
  while (error != GL_NO_ERROR) {
    std::string err = std::string("GL API ERROR :") + std::to_string(error);
    LOGFL(err, LogLevel::DEBUG, filename, "", line);
    error = glGetError();
  }
#endif
}

inline void glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {

  std::string debug = "GL API DEBUG MESSAGE :\n" + std::string("Source:") + std::to_string(source) + "\n" + std::string("Type:") +
                      std::to_string(type) + "\n" + std::string("ID:") + std::to_string(id) + "\n" + std::string("Severity:") +
                      std::to_string(severity) + "\n" + std::string("Message:") + message + "\n";
  LOGS(debug);
}

class GLDebugLogs {};

#endif
