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
class ShaderDatabase final : public IResourceDB<Shader::TYPE, Shader> {
 public:
  /**
   * @brief Construct a new Shader Database object
   *
   */
  explicit ShaderDatabase(controller::ProgressStatus *progress_manager = nullptr);

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

  bool remove(Shader::TYPE type) override;
  bool remove(const Shader *shader) override;

  /**
   * @brief Recompile the database of shaders
   *
   */
  void recompile();

  /**
   * @brief Initialize the shaders
   *
   */
  void initializeShaders();

  /**
   * @brief Add a shader into the database. Contrary to the method overriden , this version adds a shader only if it's type doesn't exist in the
   * database. In case there's already a shader of a same type , it is returned instead.
   *
   * @param shader Shader object
   * @return Shader::TYPE ID of the shader in database
   */
  database::Result<Shader::TYPE, Shader> add(std::unique_ptr<Shader> shader, bool keep) override;

  /**
   * @brief This database stores only a static amount of shaders , each unique (for now) .
   * Hence , this method will always return Shader::EMPTY , as shaders are already inserted according to their type id.
   */
  Shader::TYPE firstFreeId() const override;

 private:
};

namespace database::shader {

  template<class TYPE, class... Args>
  static database::Result<Shader::TYPE, TYPE> store(IResourceDB<Shader::TYPE, Shader> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(Shader, TYPE);
    Shader::TYPE type = TYPE::getType_static();
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
