#ifndef SHADERFACTORY_H
#define SHADERFACTORY_H
#include "Factory.h"
#include "ResourceDatabaseManager.h"
#include "Shader.h"
/**
 * @file ShaderFactory.h
 * File defining the creation system for shaders
 *
 */

/**
 * @brief Shader factory class definition
 *
 */
class ShaderBuilder {

 public:
  /**
   * @brief Constructs a shader of type "type" , using vertex_code and fragment_code
   *
   * @param vertex_code Source code of the vertex shader
   * @param fragment_code Source code of the fragment shader
   * @param type Type of the shader we want to create
   * @return Shader* Created shader
   * @see Shader::TYPE
   * @see Shader
   */
  template<class TYPE, class... Args>
  static std::unique_ptr<TYPE> build(Args &&...args) {
    ASSERT_SUBTYPE(Shader, TYPE);
    return std::make_unique<PRVINTERFACE<TYPE, Args...>>(std::forward<Args>(args)...);
  }

  template<class TYPE, class... Args>
  static database::Result<Shader::TYPE, TYPE> store(IResourceDB<Shader::TYPE, Shader> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(Shader, TYPE);
    std::unique_ptr<Shader> temp = std::make_unique<PRVINTERFACE<TYPE, Args...>>(std::forward<Args>(args)...);
    database::Result<Shader::TYPE, Shader> result = database.add(std::move(temp), keep);
    database::Result<Shader::TYPE, TYPE> cast = {result.id, static_cast<TYPE *>(result.object)};
    return cast;
  }
};

#endif
