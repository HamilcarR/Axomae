#ifndef GL_STATIC_WRAPPERS_H
#define GL_STATIC_WRAPPERS_H

#include "gl_headers.h"
#include "DebugGL.h"

inline void ax_glPolygonMode(GLenum face, GLenum mode) { GL_ERROR_CHECK(glPolygonMode(face, mode)); }
inline void ax_glDepthFunc(GLenum func) { GL_ERROR_CHECK(glDepthFunc(func)); }
inline void ax_glDepthMask(GLboolean flag) { GL_ERROR_CHECK(glDepthMask(flag)); }
inline void ax_glEnable(GLenum cap) { GL_ERROR_CHECK(glEnable(cap)); }
inline void ax_glDisable(GLenum cap) { GL_ERROR_CHECK(glDisable(cap)); }
inline void ax_glCullFace(GLenum mode) { GL_ERROR_CHECK(glCullFace(mode)); }

#endif  // GL_STATIC_WRAPPERS_H
