#include "INodeFactory.h"
#include "ShaderFactory.h"
#include "TextureFactory.h"

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

  template<class TYPE>
  database::Result<U, T> addShader(bool persistence, std::string vert_ex = "", std::string frag_ex = "") {
    database::Result<U, TYPE> result = ShaderBuilder::store<TYPE>(database, persistence, vert_ex, frag_ex);
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
namespace shader_test {
  static void fill(DatabaseBuilderTest<Shader::TYPE, Shader>::ResultList &list,
                   DatabaseBuilderTest<Shader::TYPE, Shader> &builder,
                   bool persistence,
                   int &stored) {
    list.push_back(builder.addShader<Shader>(persistence));
    list.push_back(builder.addShader<BlinnPhongShader>(persistence));
    list.push_back(builder.addShader<CubemapShader>(persistence));
    list.push_back(builder.addShader<BRDFShader>(persistence));
    list.push_back(builder.addShader<ScreenFramebufferShader>(persistence));
    list.push_back(builder.addShader<BoundingBoxShader>(persistence));
    list.push_back(builder.addShader<EnvmapCubemapBakerShader>(persistence));
    list.push_back(builder.addShader<IrradianceCubemapBakerShader>(persistence));
    list.push_back(builder.addShader<EnvmapPrefilterBakerShader>(persistence));
    list.push_back(builder.addShader<BRDFLookupTableBakerShader>(persistence));
    stored = list.size();  // Increment when adding new texture
  }

};  // namespace shader_test

TEST(ShaderDatabaseTest, add) {
  ShaderDatabase database;
  DatabaseBuilderTest<Shader::TYPE, Shader> builder(database);
  DatabaseBuilderTest<Shader::TYPE, Shader>::ResultList list;
  int i = 0;
  shader_test::fill(list, builder, false, i);
  EXPECT_EQ(builder.stored.size(), list.size());
  for (const auto &A : builder.stored) {
    EXPECT_GE(A.first, 0);
  }
}

TEST(ShaderDatabaseTest, contains) {
  ShaderDatabase database;
  DatabaseBuilderTest<Shader::TYPE, Shader> builder(database);
  DatabaseBuilderTest<Shader::TYPE, Shader>::ResultList list;
  int i = 0;
  shader_test::fill(list, builder, false, i);
  for (const auto &A : list) {
    auto it = builder.stored.find(A.id);
    EXPECT_NE(it, builder.stored.end());
    EXPECT_EQ(it->first, A.id);
  }
}

TEST(ShaderDatabaseTest, remove) {
  ShaderDatabase database;
  DatabaseBuilderTest<Shader::TYPE, Shader> builder(database);
  DatabaseBuilderTest<Shader::TYPE, Shader>::ResultList list;
  int i = 0;
  shader_test::fill(list, builder, false, i);
  for (auto &A : list) {
    EXPECT_TRUE(database.remove(A.id));
    EXPECT_FALSE(database.remove(A.id));
  }
}

TEST(ShaderDatabaseTest, get) {
  ShaderDatabase database;
  DatabaseBuilderTest<Shader::TYPE, Shader> builder(database);
  DatabaseBuilderTest<Shader::TYPE, Shader>::ResultList list;
  int i = 0;
  shader_test::fill(list, builder, false, i);
  for (auto &A : list) {
    EXPECT_EQ(database.get(A.id), builder.stored.at(A.id).get());
  }
}

/****************************************************************************************************/
// Nodes