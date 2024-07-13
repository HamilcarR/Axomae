#ifndef GL_STATIC_WRAPPERS_H
#define GL_STATIC_WRAPPERS_H
#include "init_3D.h"

#include <DebugGL.h>

inline void ax_glPolygonMode(GLenum face, GLenum mode) { GL_ERROR_CHECK(glPolygonMode(face, mode)); }
inline void ax_glDepthFunc(GLenum func) { GL_ERROR_CHECK(glDepthFunc(func)); }
inline void ax_glDepthMask(GLboolean flag) { GL_ERROR_CHECK(glDepthMask(flag)); }
inline void ax_glEnable(GLenum cap) { GL_ERROR_CHECK(glEnable(cap)); }
inline void ax_glDisable(GLenum cap) { GL_ERROR_CHECK(glDisable(cap)); }
inline void ax_glCullFace(GLenum mode) { GL_ERROR_CHECK(glCullFace(mode)); }

/*******************************************************************************************************************************/
inline void ax_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLint stride, const void *ptr) {
  GL_ERROR_CHECK(glVertexAttribPointer(index, size, type, normalized, stride, ptr));
}
inline void ax_glDeleteShader(GLuint id) { GL_ERROR_CHECK(glDeleteShader(id)); }
inline void ax_glDeleteProgram(GLuint id) { GL_ERROR_CHECK(glDeleteProgram(id)); }
inline void ax_glUseProgram(GLuint id) { GL_ERROR_CHECK(glUseProgram(id)); }
inline void ax_glUniform1i(GLint location, GLint value) { GL_ERROR_CHECK(glUniform1i(location, value)); }
inline void ax_glUniform1f(GLint location, GLfloat value) { GL_ERROR_CHECK(glUniform1f(location, value)); }
inline void ax_glUniform1ui(GLint location, GLuint value) { GL_ERROR_CHECK(glUniform1ui(location, value)); }
inline void ax_glUniformMatrix4fv(GLint location, GLint count, GLboolean transpose, const GLfloat *ptr) {
  GL_ERROR_CHECK(glUniformMatrix4fv(location, count, transpose, ptr));
}
inline void ax_glUniformMatrix3fv(GLint location, GLint count, GLboolean transpose, const GLfloat *ptr) {
  GL_ERROR_CHECK(glUniformMatrix3fv(location, count, transpose, ptr));
}
inline void ax_glUniform4f(GLint location, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { GL_ERROR_CHECK(glUniform4f(location, x, y, z, w)); }
inline void ax_glUniform3f(GLint location, GLfloat x, GLfloat y, GLfloat z) { GL_ERROR_CHECK(glUniform3f(location, x, y, z)); }
inline void ax_glUniform2f(GLint location, GLfloat x, GLfloat y) { GL_ERROR_CHECK(glUniform2f(location, x, y)); }
inline void ax_glGetShaderiv(GLuint shader, GLenum pname, GLint *params) { GL_ERROR_CHECK(glGetShaderiv(shader, pname, params)); }
inline void ax_glGetShaderInfoLog(GLuint shader, GLsizei maxLength, GLsizei *length, GLchar *infoLog) {
  GL_ERROR_CHECK(glGetShaderInfoLog(shader, maxLength, length, infoLog));
}
inline void ax_glGetProgramiv(GLuint program, GLenum pname, GLint *params) { GL_ERROR_CHECK(glGetProgramiv(program, pname, params)); }
inline void ax_glGetProgramInfoLog(GLuint program, GLsizei maxLength, GLsizei *length, GLchar *infoLog) {
  GL_ERROR_CHECK(glGetProgramInfoLog(program, maxLength, length, infoLog));
}
inline void ax_glAttachShader(GLuint program, GLuint shader) { GL_ERROR_CHECK(glAttachShader(program, shader)); }
inline void ax_glLinkProgram(GLuint program) { GL_ERROR_CHECK(glLinkProgram(program)); }
inline void ax_glCompileShader(GLuint shader) { GL_ERROR_CHECK(glCompileShader(shader)); }
inline void ax_glShaderSource(GLuint shader, GLsizei count, const GLchar **string, const GLint *length) {
  GL_ERROR_CHECK(glShaderSource(shader, count, string, length));
}
inline void ax_glEnableVertexAttribArray(GLuint attribute) { GL_ERROR_CHECK(glEnableVertexAttribArray(attribute)); }
#endif  // GL_STATIC_WRAPPERS_H
