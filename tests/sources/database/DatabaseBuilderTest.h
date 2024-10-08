#ifndef DATABASEBUILDERTEST_H
#define DATABASEBUILDERTEST_H
#include "INodeDatabase.h"
#include "ImageDatabase.h"
#include "RenderingDatabaseInterface.h"
#include "ShaderDatabase.h"
#include "Test.h"
#include "TextureDatabase.h"

static void rand_init() { srand(time(nullptr)); }

template<class U, class T>
class DatabaseBuilderTest {
 public:
  using ResultList = std::vector<database::Result<U, T>>;
  using StorageMap = std::map<U, database::Storage<U, T>>;
  explicit DatabaseBuilderTest(IResourceDB<U, T> &DB) : database(DB), stored(database.getConstData()) {}
  ~DatabaseBuilderTest() = default;
  DatabaseBuilderTest(const DatabaseBuilderTest &) = delete;
  DatabaseBuilderTest(DatabaseBuilderTest &&) = delete;
  DatabaseBuilderTest &operator=(DatabaseBuilderTest &&) = delete;
  DatabaseBuilderTest &operator=(const DatabaseBuilderTest &) = delete;

  int getDatabaseSize() { return database.size(); }
  const StorageMap &getStoredDatabase() { return stored; }

 public:
  IResourceDB<U, T> &database{};
  const StorageMap &stored{};
};

#endif