#include "ShaderDatabase.h"
#include "Mutex.h"
#include "ShaderFactory.h"

ShaderDatabase::ShaderDatabase() {}

void ShaderDatabase::purge() {
  for (auto &A : database)
    A.second->clean();
  database.clear();
}

void ShaderDatabase::clean() {
  /*The shader database doesn't need a special clean-up method for now, we just load any shader needed and purge the database at the end of the
   * program */
}

void ShaderDatabase::initializeShaders() {
  Mutex::Lock lock(mutex);
  for (auto &A : database)
    A.second->initializeShader();
}

Shader *ShaderDatabase::get(const Shader::TYPE type) const {
  Mutex::Lock lock(mutex);
  auto it = database.find(type);
  return it == database.end() ? nullptr : it->second.get();
}

void ShaderDatabase::recompile() {
  Mutex::Lock lock(mutex);
  for (auto &A : database)
    A.second->recompile();
}

database::Result<Shader::TYPE, Shader> ShaderDatabase::add(std::unique_ptr<Shader> shader, bool /*keep*/) {
  Mutex::Lock lock(mutex);
  for (auto &A : database)
    if (A.first == shader->getType())
      return {A.first, A.second.get()};
  Shader::TYPE id = shader->getType();
  database[id] = std::move(shader);
  return {id, database[id].get()};
}

database::Result<Shader::TYPE, Shader> ShaderDatabase::contains(const Shader *shader) const {
  Mutex::Lock lock(mutex);
  for (const auto &A : database) {
    if (A.second.get() == shader)
      return {A.first, A.second.get()};
  }
  return {Shader::EMPTY, nullptr};
}

bool ShaderDatabase::contains(const Shader::TYPE type) const {
  Mutex::Lock lock(mutex);
  return database.find(type) != database.end();
}

bool ShaderDatabase::remove(const Shader *shader) {
  for (auto it = database.begin(); it != database.end(); it++) {
    if (shader == it->second.get()) {
      it->second.get()->clean();
      database.erase(it);
    }
    return true;
  }
  return false;
}

bool ShaderDatabase::remove(const Shader::TYPE type) {
  auto shader = get(type);
  if (!shader)
    return false;
  shader->clean();
  database.erase(type);
  return true;
}