
#include "../texture/texture_list.h"
#include "DatabaseBuilderTest.h"

const int COUNT = 16;
namespace texture_database_test {

  template<class HEAD, class... TAIL>
  constexpr void addTexture(IResourceDB<int, GenericTexture> &database, TextureData *data) {
    bool persistence = math::random::randb();
    database::texture::store<HEAD>(database, persistence, data);
    if constexpr (sizeof...(TAIL) > 0)
      addTexture<TAIL...>(database, data);
  }
}  // namespace texture_database_test

class TextureDatabaseTest final : public DatabaseBuilderTest<int, GenericTexture> {
 public:
  explicit TextureDatabaseTest(IResourceDB<int, GenericTexture> &db, int total) : DatabaseBuilderTest<int, GenericTexture>(db) {
    total = (total < COUNT) ? COUNT : total;
    buildDatabase();
  }
  explicit TextureDatabaseTest(IResourceDB<int, GenericTexture> &db) : DatabaseBuilderTest<int, GenericTexture>(db) { buildDatabase(); }
  template<class TYPE, class... Args>
  database::Result<int, GenericTexture> add(bool persistence, Args &&...args) {
    auto result = database::texture::store<TYPE>(database, persistence, std::forward<Args>(args)...);
    database::Result<int, GenericTexture> cast = {result.id, static_cast<GenericTexture *>(result.object)};
    return cast;
  }

  int getPersistentSize() {
    int pers = 0;
    for (auto &elem : stored) {
      if (elem.second.isPersistent())
        pers++;
    }
    return pers;
  }

  int getNonPersistentSize() {
    int no_pers = 0;
    for (auto &elem : stored) {
      if (!elem.second.isPersistent())
        no_pers++;
    }
    return no_pers;
  }

 private:
  void buildDatabase() {
    TextureData data;
    texture_database_test::addTexture<TYPE_LIST>(database, &data);
    total_size = database.size();
    number_persistent = getPersistentSize();
  }

 public:
  int total_size{0};
  int number_persistent{0};
};

TEST(TextureDatabaseTest, add) {
  TextureDatabase database;
  TextureDatabaseTest test(database);
  /* Check if all texture types have been created and stored*/
  int sizeDB = test.getDatabaseSize();
  int incremented = test.total_size;
  EXPECT_EQ(sizeDB, incremented);
  /* Textures with a non-empty name should be unique*/
  TextureData data1, data2;
  data1.name = std::string("texture1");
  data2.name = std::string("texture2");
  auto result1 = test.add<DiffuseTexture>(false, &data1);
  auto result2 = test.add<DiffuseTexture>(false, &data2);
  EXPECT_NE(result1, result2);
  TextureData data3;
  data3.name = std::string("texture1");
  auto result3 = test.add<DiffuseTexture>(false, &data3);
  EXPECT_EQ(result1, result3);

  /* Dummy textures should be unique */
  result1 = test.add<SpecularTexture>(false, nullptr);
  result2 = test.add<SpecularTexture>(false, nullptr);
  result3 = test.add<MetallicTexture>(false, nullptr);
  EXPECT_EQ(result1, result2);
  EXPECT_NE(result1, result3);
}

TEST(TextureDatabaseTest, contains) {
  TextureDatabase database;
  TextureDatabaseTest test(database);
  EXPECT_EQ(database.contains(nullptr).object, nullptr);
  EXPECT_EQ(database.contains(database.size() + 1), false);
}

TEST(TextureDatabaseTest, remove) {
  TextureDatabase database;
  TextureDatabaseTest test(database);
  EXPECT_FALSE(database.remove(nullptr));
  for (int i = 0; i < COUNT; i++) {
    int pos = math::random::nrandi(0, COUNT - 1);
    auto iterator = database.getConstData().find(pos);
    bool contain = iterator != database.getConstData().end();
    EXPECT_EQ(contain, database.remove(iterator->second.get()));
  }
}

TEST(TextureDatabaseTest, get) {
  TextureDatabase database;
  TextureDatabaseTest test(database);
  std::vector<GenericTexture *> ptr_list;
  for (const auto &elem : database.getConstData())
    ptr_list.push_back(elem.second.get());
  /* Checks that the database isn't modified */
  int original_size = database.size();
  for (int i = 0; i < database.size(); i++) {
    database.get(i);
    EXPECT_EQ(database.size(), original_size);
  }
  EXPECT_EQ(ptr_list.size(), database.size());
  int i = 0;
  for (const auto &elem : ptr_list) {
    EXPECT_EQ(database.get(i), elem);
    i++;
  }
  GenericTexture *ptr = database.get(-1);
  EXPECT_EQ(ptr, nullptr);
  ptr = database.get(database.size() + 1);
  EXPECT_EQ(ptr, nullptr);
  int pos = math::random::nrandi(0, COUNT - 1);
  ptr = database.get(pos);
  i = 0;
  bool found = false;
  for (const auto &elem : database.getConstData()) {
    if (elem.first == pos) {
      EXPECT_EQ(elem.second.get(), ptr);
      found = true;
    }
  }
  EXPECT_EQ(found, true);
}

TEST(TextureDatabaseTest, clean) {
  TextureDatabase database;
  TextureDatabaseTest test(database);
  int persists = test.number_persistent;
  database.clean();
  int final_size = database.size();
  EXPECT_EQ(final_size, persists);
}

TEST(TextureDatabaseTest, firstFreeId) {
  TextureDatabase database;
  TextureDatabaseTest test(database);
  auto id = database.firstFreeId();
  EXPECT_EQ(id, COUNT);
}