#include "ShaderDatabase.h"
#include "Mutex.h"

void ShaderDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second.get()->clean();
  database_map.clear();
}

void ShaderDatabase::clean() {
  std::vector<Shader::TYPE> to_erase;
  for (auto &elem : database_map) {
    if (!elem.second.isPersistent()) {
      elem.second.get()->clean();
      to_erase.push_back(elem.first);
    }
  }
  for (auto &elem : to_erase)
    database_map.erase(elem);
}

void ShaderDatabase::initializeShaders() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second.get()->initializeShader();
}

void ShaderDatabase::recompile() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second.get()->recompile();
}

database::Result<Shader::TYPE, Shader> ShaderDatabase::add(std::unique_ptr<Shader> shader, bool keep) {
  Mutex::Lock lock(mutex);
  DATABASE::iterator it;
  if ((it = database_map.find(shader->getType())) != database_map.end()) {
    assert(it->second.isValid());
    return {it->first, it->second.get()};
  } else {
    Shader::TYPE type = shader->getType();
    Shader *ptr = shader.get();
    database::Storage<Shader::TYPE, Shader> database(std::move(shader), shader->getType(), keep);
    database_map[type] = std::move(database);
    return {type, ptr};
  }
}

bool ShaderDatabase::remove(const Shader *shader) {
  Mutex::Lock lock(mutex);
  for (auto it = database_map.begin(); it != database_map.end(); it++) {
    if (shader == it->second.get()) {
      it->second.get()->clean();
      database_map.erase(it);
      return true;
    }
  }
  return false;
}

Shader::TYPE ShaderDatabase::firstFreeId() const { return Shader::EMPTY; }

bool ShaderDatabase::remove(const Shader::TYPE type) {
  auto shader = get(type);
  if (!shader)
    return false;
  Mutex::Lock lock(mutex);
  shader->clean();
  database_map.erase(type);
  return true;
}