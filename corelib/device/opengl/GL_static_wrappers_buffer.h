

#ifndef GL_STATIC_WRAPPERS_BUFFER_H
#define GL_STATIC_WRAPPERS_BUFFER_H
#include "DebugGL.h"
#include "gl_headers.h"

inline void ax_glGenBuffers(GLsizei n, GLuint *buffers) { GL_ERROR_CHECK(glGenBuffers(n, buffers)); }
inline void ax_glBindBuffer(GLenum target, GLuint id) { GL_ERROR_CHECK(glBindBuffer(target, id)); }
inline void ax_glDeleteBuffers(GLsizei n, GLuint *buffers) { GL_ERROR_CHECK(glDeleteBuffers(n, buffers)); }
inline void ax_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
  GL_ERROR_CHECK(glBufferData(target, size, data, usage));
}
inline void ax_glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
  GL_ERROR_CHECK(glBufferSubData(target, offset, size, data));
}
inline GLboolean ax_glUnmapBuffer(GLenum target) {
  bool b = glUnmapBuffer(target);
  errorCheck(__FILE__, __func__, __LINE__);
  return b;
}

inline void ax_glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
  GL_ERROR_CHECK(glFlushMappedBufferRange(target, offset, length));
}

inline void *ax_glMapBuffer(GLenum target, GLenum access) {
  void *ret = glMapBuffer(target, access);
  errorCheck(__FILE__, __func__, __LINE__);
  return ret;
}

inline void *ax_glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
  void *ret = glMapBufferRange(target, offset, length, access);
  errorCheck(__FILE__, __func__, __LINE__);
  return ret;
}

inline void ax_glGenVertexArrays(GLsizei n, GLuint *arrays) { GL_ERROR_CHECK(glGenVertexArrays(n, arrays)); }
inline void ax_glBindVertexArray(GLuint array) { GL_ERROR_CHECK(glBindVertexArray(array)); }
inline void ax_glDeleteVertexArrays(GLsizei n, GLuint *buffers) { GL_ERROR_CHECK(glDeleteVertexArrays(n, buffers)); }
#endif  // GL_STATIC_WRAPPERS_BUFFER_H
