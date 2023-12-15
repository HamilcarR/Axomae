#include "../includes/INodeFactory.h"
#include "../includes/ShaderFactory.h"
#include "../includes/TextureFactory.h"

#include "Test.h"

static void rand_init() { srand(time(nullptr)); }

template<class U, class T>
class DatabaseBuilderTest {
 public:
  using ResultList = std::vector<database::Result<U, T>>;

  DatabaseBuilderTest(IResourceDB<U, T> &DB) : database(DB), stored(database.getConstData()) { rand_init(); }

  template<class TYPE>
  database::Result<U, T> addTexture(bool persistence, TextureData *data) {
    database::Result<U, TYPE> result = TextureBuilder::store<TYPE>(database, persistence, data);
    database::Result<U, T> cast = {result.id, static_cast<T *>(result.object)};
    return cast;
  }

  IResourceDB<U, T> &database;
  const std::map<U, std::unique_ptr<T>> &stored;
};

/****************************************************************************************************/
// Textures
/*Non persistent textures*/
namespace texture_test {

  static void fill(DatabaseBuilderTest<int, Texture>::ResultList &list,
                   DatabaseBuilderTest<int, Texture> &texbuilder,
                   bool persistence,
                   TextureData &data,
                   int &stored) {
    list.push_back(texbuilder.addTexture<DiffuseTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<NormalTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<MetallicTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<RoughnessTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<AmbiantOcclusionTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<SpecularTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<EmissiveTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<OpacityTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<CubemapTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<EnvironmentMap2DTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<IrradianceTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<BRDFLookupTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<FrameBufferTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<GenericCubemapTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<Generic2DTexture>(persistence, &data));
    list.push_back(texbuilder.addTexture<RoughnessTexture>(persistence, &data));
    stored = list.size();  // Increment when adding new texture
  }

};  // namespace texture_test

TEST(TextureDatabaseTest, add) {
  TextureDatabase database;
  DatabaseBuilderTest<int, Texture> texbuilder(database);
  DatabaseBuilderTest<int, Texture>::ResultList list;
  TextureData data;
  int size;  // number of textures stored
  texture_test::fill(list, texbuilder, false, data, size);
  EXPECT_EQ(texbuilder.stored.size(), size);  // Check if textures have been created and stored separately (no old texture returned)
  for (auto &stored : texbuilder.stored)
    EXPECT_GE(stored.first, 0);
  list.clear();
  TextureDatabase database2;
  DatabaseBuilderTest<int, Texture> texbuilder2(database2);
  texture_test::fill(list, texbuilder2, true, data, size);
  for (auto &stored : texbuilder2.stored) {
    EXPECT_LE(stored.first, -1);
  }
}

TEST(TextureDatabaseTest, contains) {
  TextureDatabase database;
  DatabaseBuilderTest<int, Texture> texbuilder(database);
  DatabaseBuilderTest<int, Texture>::ResultList list;
  TextureData data;
  int size;
  texture_test::fill(list, texbuilder, false, data, size);
  for (auto &stored : texbuilder.stored) {
    database::Result<int, Texture> result = texbuilder.database.contains(stored.second.get());
    EXPECT_EQ(result.object, stored.second.get());
    EXPECT_EQ(result.id, stored.first);
  }
  database::Result<int, Texture> result = texbuilder.database.contains(nullptr);
  EXPECT_EQ(result.object, nullptr);
}

TEST(TextureDatabaseTest, remove) {
  TextureDatabase database;
  DatabaseBuilderTest<int, Texture> texbuilder(database);
  DatabaseBuilderTest<int, Texture>::ResultList list;
  TextureData data;
  int size;
  texture_test::fill(list, texbuilder, false, data, size);
  bool test = texbuilder.database.remove(list.size() + 1);
  EXPECT_FALSE(test);
  test = texbuilder.database.remove(nullptr);
  EXPECT_FALSE(test);
  for (auto elem : list)
    EXPECT_TRUE(texbuilder.database.remove(elem.id));
}

TEST(TextureDatabaseTest, get) {
  TextureDatabase database;
  DatabaseBuilderTest<int, Texture> texbuilder(database);
  DatabaseBuilderTest<int, Texture>::ResultList list;
  TextureData data;
  int size;
  texture_test::fill(list, texbuilder, false, data, size);
  for (auto elem : list) {
    Texture *tex = texbuilder.database.get(elem.id);
    EXPECT_EQ(tex, texbuilder.stored.at(elem.id).get());
  }
}

/****************************************************************************************************/
// Shaders

/****************************************************************************************************/
// Nodes