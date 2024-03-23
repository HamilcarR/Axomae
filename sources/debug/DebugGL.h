#ifndef DEBUGGL_H
#define DEBUGGL_H
#include "Logger.h"
#include "constants.h"
#include "init_3D.h"
#include <iostream>
#include <sstream>

inline void errorCheck(const char *filename, unsigned int line) {
#ifndef NDEBUG
  GLenum error = GL_NO_ERROR;
  error = glGetError();
  if (error != GL_NO_ERROR) {
    std::string err = std::string("GL API ERROR :") + std::to_string(error);
    LOG(err, LogLevel::ERROR);
  }
#endif
}

inline void glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
  std::stringstream str;
  str << "OpenGL Debug Message:"
      << "Source:" << source << "\n"
      << "Type:" << type << "\n"
      << "ID:" << id << "\n"
      << "Severity:" << severity << "\n"
      << "Message:" << message << "\n";
  std::string debug = "GL API DEBUG MESSAGE :\n" + std::string("Source:") + std::to_string(source) + "\n" + std::string("Type:") +
                      std::to_string(type) + "\n" + std::string("ID:") + std::to_string(id) + "\n" + std::string("Severity:") +
                      std::to_string(severity) + "\n" + std::string("Message:") + message + "\n";
  LOG(debug, LogLevel::GLINFO);
}

class GLDebugLogs {};

#endif
