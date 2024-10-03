#ifndef GL_STATIC_WRAPPERS_TEXTURES_H
#define GL_STATIC_WRAPPERS_TEXTURES_H
#include "DebugGL.h"
#include "gl_headers.h"

inline void ax_glTexSubImage2D(
    GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
  GL_ERROR_CHECK(glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels));
}
inline void ax_glTextureSubImage2D(
    GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {
  GL_ERROR_CHECK(glTexSubImage2D(texture, level, xoffset, yoffset, width, height, format, type, pixels));
}
inline void ax_glGenTextures(GLuint n, GLuint *textures) { GL_ERROR_CHECK(glGenTextures(n, textures)); }
inline void ax_glActiveTexture(GLenum texture) { GL_ERROR_CHECK(glActiveTexture(texture)); }
inline void ax_glBindTexture(GLenum texture, GLuint id) { GL_ERROR_CHECK(glBindTexture(texture, id)); }
inline void ax_glTexImage2D(
    GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *data) {
  GL_ERROR_CHECK(glTexImage2D(target, level, internalformat, width, height, border, format, type, data));
}
inline void ax_glTexParameteri(GLenum target, GLenum pname, GLint param) { GL_ERROR_CHECK(glTexParameteri(target, pname, param)); }

inline void ax_glGenerateMipmap(GLenum target) { GL_ERROR_CHECK(glGenerateMipmap(target)); }

#endif  // GL_STATIC_WRAPPERS_TEXTURES_H
