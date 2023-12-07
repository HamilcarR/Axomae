#ifndef SHADERDATABASE_H
#define SHADERDATABASE_H
#include "RenderingDatabaseInterface.h"
#include "Shader.h"
#include "ShaderFactory.h"
#include "utils_3D.h"
#include <map>

/**
 * @file ShaderDatabase.h
 * A database containing Shader pointers to avoid duplicating and copying Shaders with the same informations
 *
 */

/**
 * @brief ShaderDatabase class implementation
 *
 */
class ShaderDatabase : public RenderingDatabaseInterface<Shader::TYPE, Shader> {
 public:
  /**
   * @brief Construct a new Shader Database object
   *
   */
  ShaderDatabase();

  /**
   * @brief Destroy the Shader Database object
   *
   */
  virtual ~ShaderDatabase();

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
   * @brief This function constructs a shader and stores it in the shader database if it does not already exist and
   * returns it.
   *
   * @param vertex_code A string containing the source code for the vertex shader.
   * @param fragment_code A string containing the source code for the fragment shader.
   *
   * @return Shader* Pointer to the constructed shader , or the existing one
   *
   */
  template<class TYPE>
  Shader *addShader(const std::string vertex_code, const std::string fragment_code) {
    Mutex::Lock lock(mutex);
    std::unique_ptr<TYPE> temp_shader = ShaderBuilder::build<TYPE>(vertex_code, fragment_code);
    auto type = temp_shader->getType();
    if (shader_database.find(type) == shader_database.end())
      shader_database[type] = std::move(temp_shader);
    return shader_database[type].get();
  }

  /**
   * The function checks if a shader type exists in a shader database.
   *
   * @param type The parameter "type" is of type Shader::TYPE, which is an enumerated type representing
   * different types of shaders (e.g. vertex shader, fragment shader, geometry shader, etc.). It is used
   * to look up a shader in the shader_database map.
   *
   * @return The function `contains` returns a boolean value indicating whether the `shader_database`
   * contains a shader of the specified `type`.
   */
  virtual bool contains(const Shader::TYPE type) override;

  /**
   * This function returns a pointer to a shader object of a given type from a shader database, or
   * nullptr if it does not exist.
   *
   * @param type The parameter "type" is of type Shader::TYPE, which is an enumerated type representing
   * different types of shaders (e.g. vertex shader, fragment shader, geometry shader, etc.). It is used
   * to look up a shader in the shader_database map.
   *
   * @return a pointer to a Shader object of the specified type if it exists in the shader_database map.
   * If the shader of the specified type does not exist in the map, the function returns a null pointer.
   */
  Shader *get(const Shader::TYPE type) override;

  /**
   * @brief
   *
   * @param type
   * @return true
   * @return false
   */
  virtual bool remove(const Shader::TYPE type);

  /**
   * @brief
   *
   * @param shader
   * @return true
   * @return false
   */
  virtual bool remove(const Shader *shader);

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
  virtual Shader::TYPE add(std::unique_ptr<Shader> shader, bool keep);

  /**
   * @brief Checks if the database contains this shader . returns a pair of it's ID and it's address
   *
   * @param shader Shader to search
   * @return std::pair<Shader::TYPE , Shader*> Pair <ID , Shader*>. If nothing found , returns <Shader::EMPTY , nullptr>
   */
  virtual std::pair<Shader::TYPE, Shader *> contains(const Shader *shader) override;

 private:
  std::map<Shader::TYPE, std::unique_ptr<Shader>> shader_database;
};

#endif
