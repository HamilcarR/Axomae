#ifndef SHADERDATABASE_H
#define SHADERDATABASE_H
#include "Factory.h"
#include "RenderingDatabaseInterface.h"
#include "Shader.h"
/**
 * @file ShaderDatabase.h
 * A database containing Shader pointers to avoid duplicating and copying Shaders with the same informations
 *
 */

/**
 * @brief ShaderDatabase class implementation
 *
 */
class ShaderDatabase : public IResourceDB<Shader::TYPE, Shader> {
 public:
  /**
   * @brief Construct a new Shader Database object
   *
   */
  ShaderDatabase();

  /**
   * @brief Cleans the whole database , Deletes all shaders .
   *
   */
  void clean() override;

  /**
   * @brief
   *
   */
  void purge() override;

  /**
   * This function returns a pointer to a shader object of a given type from a shader database, or
   * nullptr if it does not exist.
   *
   * @param type The parameter "type" is of type Shader::TYPE, which is an enumerated type representing
   * different types of shaders (e.g. vertex shader, fragment shader, geometry shader, etc.). It is used
   * to look up a shader in the database map.
   *
   * @return a pointer to a Shader object of the specified type if it exists in the database map.
   * If the shader of the specified type does not exist in the map, the function returns a null pointer.
   */
  Shader *get(const Shader::TYPE type) const override;
  virtual bool remove(const Shader::TYPE type) override;
  virtual bool remove(const Shader *shader) override;

  /**
   * @brief Recompile the database of shaders
   *
   */
  virtual void recompile();

  /**
   * @brief Initialize the shaders
   *
   */
  virtual void initializeShaders();

  /**
   * @brief Add a shader into the database
   *
   * @param shader Shader object
   * @param keep Not used ... for now .
   * @return Shader::TYPE ID of the shader in database
   */
  virtual database::Result<Shader::TYPE, Shader> add(std::unique_ptr<Shader> shader, bool keep);

  const std::map<Shader::TYPE, std::unique_ptr<Shader>> &getConstData() const override { return database_map; }

 private:
};

namespace database::shader {

  template<class TYPE, class... Args>
  static database::Result<Shader::TYPE, TYPE> store(IResourceDB<Shader::TYPE, Shader> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(Shader, TYPE);
    constexpr Shader::TYPE type = shader_utils::get_type<TYPE>();
    Shader *seek = database.get(type);
    if (seek) {
      database::Result<Shader::TYPE, TYPE> result = {type, static_cast<TYPE *>(seek)};
      return result;
    }
    std::unique_ptr<Shader> temp = std::make_unique<PRVINTERFACE<TYPE, Args...>>(std::forward<Args>(args)...);
    database::Result<Shader::TYPE, Shader> result = database.add(std::move(temp), keep);
    database::Result<Shader::TYPE, TYPE> cast = {result.id, static_cast<TYPE *>(result.object)};
    return cast;
  }
};  // namespace database::shader

#endif
