#include "ShaderDatabase.h"
#include "Mutex.h"
#include "ShaderFactory.h"

ShaderDatabase::ShaderDatabase() {}

void ShaderDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second->clean();
  database_map.clear();
}

void ShaderDatabase::clean() {
  // The shader database doesn't need a special clean-up method for now, we just load any shader needed and purge the database at the end of the
  // program
  return;
}

void ShaderDatabase::initializeShaders() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second->initializeShader();
}

Shader *ShaderDatabase::get(const Shader::TYPE type) const {
  Mutex::Lock lock(mutex);
  auto it = database_map.find(type);
  return it == database_map.end() ? nullptr : it->second.get();
}

void ShaderDatabase::recompile() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second->recompile();
}

database::Result<Shader::TYPE, Shader> ShaderDatabase::add(std::unique_ptr<Shader> shader, bool /*keep*/) {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    if (A.first == shader->getType())
      return {A.first, A.second.get()};
  Shader::TYPE id = shader->getType();
  database_map[id] = std::move(shader);
  return {id, database_map[id].get()};
}

bool ShaderDatabase::remove(const Shader *shader) {
  for (auto it = database_map.begin(); it != database_map.end(); it++) {
    if (shader == it->second.get()) {
      it->second.get()->clean();
      database_map.erase(it);
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
  database_map.erase(type);
  return true;
}