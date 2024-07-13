#ifndef GL_STATIC_WRAPPERS_FRAMEBUFFER_H
#define GL_STATIC_WRAPPERS_FRAMEBUFFER_H

#include "DebugGL.h"
#include "gl_headers.h"

inline void ax_glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
  GL_ERROR_CHECK(glFramebufferTexture2D(target, attachment, textarget, texture, level));
}
inline void ax_glGenFramebuffers(GLsizei n, GLuint *framebuffers) { GL_ERROR_CHECK(glGenFramebuffers(n, framebuffers)); }
inline void ax_glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
  GL_ERROR_CHECK(glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer));
}
inline void ax_glBindFramebuffer(GLenum target, GLuint framebuffer) { GL_ERROR_CHECK(glBindFramebuffer(target, framebuffer)); }

inline void ax_glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) { GL_ERROR_CHECK(glDeleteFramebuffers(n, framebuffers)); }

inline void ax_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) { GL_ERROR_CHECK(glGenRenderbuffers(n, renderbuffers)); }
inline void ax_glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {
  GL_ERROR_CHECK(glRenderbufferStorage(target, internalformat, width, height));
}

inline void ax_glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffer) { GL_ERROR_CHECK(glDeleteRenderbuffers(n, renderbuffer)); }

inline void ax_glBindRenderbuffer(GLenum target, GLuint renderbuffer) { GL_ERROR_CHECK(glBindRenderbuffer(target, renderbuffer)); }
#endif  // GL_STATIC_WRAPPERS_FRAMEBUFFER_H
