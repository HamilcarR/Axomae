#include "../includes/ShaderDatabase.h"
#include "../includes/Mutex.h"
#include "../includes/ShaderFactory.h"

ShaderDatabase::ShaderDatabase() {}

ShaderDatabase::~ShaderDatabase() {}

void ShaderDatabase::purge() {
  clean();
}

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

bool ShaderDatabase::contains(const Shader::TYPE type) {
  Mutex::Lock lock(mutex);
  return shader_database.find(type) != shader_database.end();
}

Shader *ShaderDatabase::get(const Shader::TYPE type) {
  Mutex::Lock lock(mutex);
  auto it = shader_database.find(type);
  if (it != shader_database.end())
    return it->second.get();
  else
    return nullptr;
}

void ShaderDatabase::recompile() {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database)
    A.second->recompile();
}

Shader::TYPE ShaderDatabase::add(std::unique_ptr<Shader> shader, bool keep) {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database)
    if (A.second.get() == shader.get())
      return A.first;
  shader_database[shader->getType()] = std::move(shader);
  return shader->getType();
}

std::pair<Shader::TYPE, Shader *> ShaderDatabase::contains(const Shader *shader) {
  Mutex::Lock lock(mutex);
  for (auto &A : shader_database) {
    if (A.second.get() == shader)
      return std::pair(A.first, A.second.get());
  }
  return std::pair(Shader::EMPTY, nullptr);
}

bool ShaderDatabase::remove(const Shader *shader) {
  return false;
}

bool ShaderDatabase::remove(const Shader::TYPE type) {
  return false;
}