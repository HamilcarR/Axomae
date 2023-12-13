#include "../includes/ShaderDatabase.h"
#include "../includes/Mutex.h"
#include "../includes/ShaderFactory.h"

ShaderDatabase::ShaderDatabase() {}

void ShaderDatabase::purge() { clean(); }

void ShaderDatabase::clean() {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database) {
    A.second->clean();
    A.second = nullptr;
  }
  shader_database.clear();
}

void ShaderDatabase::initializeShaders() {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database)
    A.second->initializeShader();
}

bool ShaderDatabase::contains(const Shader::TYPE type) const {
  Mutex::Lock lock(mutex);
  return shader_database.find(type) != shader_database.end();
}

Shader *ShaderDatabase::get(const Shader::TYPE type) const {
  Mutex::Lock lock(mutex);
  auto it = shader_database.find(type);
  return it == shader_database.end() ? nullptr : it->second.get();
}

void ShaderDatabase::recompile() {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database)
    A.second->recompile();
}

database::Result<Shader::TYPE, Shader> ShaderDatabase::add(std::unique_ptr<Shader> shader, bool keep) {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database)
    if (A.second.get() == shader.get())
      return {A.first, A.second.get()};
  Shader::TYPE id = shader->getType();
  shader_database[id] = std::move(shader);
  return {id, shader_database[id].get()};
}

database::Result<Shader::TYPE, Shader> ShaderDatabase::contains(const Shader *shader) const {
  Mutex::Lock lock(mutex);
  for (const auto &A : shader_database) {
    if (A.second.get() == shader)
      return {A.first, A.second.get()};
  }
  return {Shader::EMPTY, nullptr};
}

bool ShaderDatabase::remove(const Shader *shader) {
  for (auto it = shader_database.begin(); it != shader_database.end(); it++) {
    if (shader == it->second.get())
      shader_database.erase(it);
    return true;
  }
  return false;
}

bool ShaderDatabase::remove(const Shader::TYPE type) { return shader_database.erase(type) != 0; }