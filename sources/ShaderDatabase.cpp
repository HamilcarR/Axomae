#include "../includes/ShaderDatabase.h"
#include "../includes/Mutex.h"
#include "../includes/ShaderFactory.h"

ShaderDatabase::ShaderDatabase() {}

void ShaderDatabase::purge() { clean(); }

void ShaderDatabase::clean() {
  Mutex::Lock lock(mutex);
  for (auto &A : database) {
    A.second->clean();
    A.second = nullptr;
  }
  database.clear();
}

void ShaderDatabase::initializeShaders() {
  Mutex::Lock lock(mutex);
  for (auto &A : database)
    A.second->initializeShader();
}

bool ShaderDatabase::contains(const Shader::TYPE type) const {
  Mutex::Lock lock(mutex);
  return database.find(type) != database.end();
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

database::Result<Shader::TYPE, Shader> ShaderDatabase::add(std::unique_ptr<Shader> shader, bool keep) {
  Mutex::Lock lock(mutex);
  for (auto &A : database)
    if (A.second.get() == shader.get())
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

bool ShaderDatabase::remove(const Shader *shader) {
  for (auto it = database.begin(); it != database.end(); it++) {
    if (shader == it->second.get())
      database.erase(it);
    return true;
  }
  return false;
}

bool ShaderDatabase::remove(const Shader::TYPE type) { return database.erase(type) != 0; }