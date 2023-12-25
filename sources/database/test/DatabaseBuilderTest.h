#ifndef DATABASEBUILDERTEST_H
#define DATABASEBUILDERTEST_H
#include "INodeDatabase.h"
#include "ImageDatabase.h"
#include "ShaderDatabase.h"
#include "Test.h"
#include "TextureDatabase.h"

static void rand_init() { srand(time(nullptr)); }

template<class U, class T>
class DatabaseBuilderTest {
 public:
  using ResultList = std::vector<database::Result<U, T>>;

  DatabaseBuilderTest(IResourceDB<U, T> &DB) : database(DB), stored(database.getConstData()) { rand_init(); }

  template<class TYPE>
  database::Result<U, T> addTexture(bool persistence, TextureData *data) {
    database::Result<U, TYPE> result = database::texture::store<TYPE>(database, persistence, data);
    database::Result<U, T> cast = {result.id, static_cast<T *>(result.object)};
    return cast;
  }

  template<class TYPE>
  database::Result<U, T> addShader(bool persistence) {
    database::Result<U, TYPE> result = database::shader::store<TYPE>(database, persistence);
    database::Result<U, T> cast = {result.id, static_cast<T *>(result.object)};
    return cast;
  }

  template<class TYPE>
  database::Result<U, T> addNode(bool persistence) {
    database::Result<U, TYPE> result = database::node::store<TYPE>(database, persistence);
    database::Result<U, T> cast = {result.id, static_cast<T *>(result.object)};
    return cast;
  }

  IResourceDB<U, T> &database;
  const std::map<U, std::unique_ptr<T>> &stored;
};
#endif