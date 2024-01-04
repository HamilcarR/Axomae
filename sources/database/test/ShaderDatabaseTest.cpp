
#include "DatabaseBuilderTest.h"
#define SHADERTYPE_LIST \
  Shader, BlinnPhongShader, CubemapShader, BRDFShader, ScreenFramebufferShader, BoundingBoxShader, EnvmapCubemapBakerShader, \
      IrradianceCubemapBakerShader, EnvmapPrefilterBakerShader, BRDFLookupTableBakerShader

/* Add to count when adding a new type of shaders */
const int COUNT = 10;
namespace shader_database_test {
  using ResultList = DatabaseBuilderTest<Shader::TYPE, Shader>::ResultList;

  template<class HEAD, class... TAIL>
  void addShader(IResourceDB<Shader::TYPE, Shader> &database, ResultList &store_list) {
    bool persistence = random_math::randb();
    auto result = database::shader::store<HEAD>(database, persistence);
    database::Result<Shader::TYPE, Shader> cast = {result.id, static_cast<Shader *>(result.object)};
    store_list.push_back(cast);
    if constexpr (sizeof...(TAIL) > 0)
      addShader<TAIL...>(database, store_list);
  }
}  // namespace shader_database_test

class ShaderDatabaseTest final : public DatabaseBuilderTest<Shader::TYPE, Shader> {
 public:
  explicit ShaderDatabaseTest(IResourceDB<Shader::TYPE, Shader> &db) : DatabaseBuilderTest<Shader::TYPE, Shader>(db) { buildDatabase(); }

 private:
  void buildDatabase() {
    ResultList list;
    shader_database_test::addShader<SHADERTYPE_LIST>(database, list);
  }
};

TEST(ShaderDatabaseTest, add) {
  ShaderDatabase database;
  std::unique_ptr<Shader> var = std::make_unique<PRVINTERFACE<BRDFShader>>();
  database.add(std::move(var), true);
  ASSERT_EQ(database.getConstData().size(), 1);
  EXPECT_TRUE(database.getConstData().at(BRDFShader::getType_static()).isValid());
  ShaderDatabaseTest test(database);
  shader_database_test::ResultList list;
  /* Checks that the database doesn't allow duplicates*/
  shader_database_test::addShader<SHADERTYPE_LIST>(database, list);
  EXPECT_EQ(list.size(), database.size());
}

/* Tests if database contains a unique shader of type HEAD */
template<class HEAD, class... TAIL>
void test_all_cases_contain(const ShaderDatabaseTest &test) {
  const Shader *ptr = test.stored.at(HEAD::getType_static()).get();
  EXPECT_EQ(test.database.contains(ptr).object, ptr);
  if constexpr (sizeof...(TAIL) > 0)
    test_all_cases_contain<TAIL...>(test);
}

TEST(ShaderDatabaseTest, contains) {
  ShaderDatabase database;
  ShaderDatabaseTest test(database);
  EXPECT_EQ(database.contains(nullptr).object, nullptr);
  test_all_cases_contain<SHADERTYPE_LIST>(test);
}

template<class HEAD, class... TAIL>
void test_all_cases_remove(const ShaderDatabaseTest &test) {
  test.database.remove(HEAD::getType_static());
  EXPECT_FALSE(test.database.remove(HEAD::getType_static()));
  if constexpr (sizeof...(TAIL) > 0)
    test_all_cases_contain<TAIL...>(test);
}

TEST(ShaderDatabaseTest, remove) {
  ShaderDatabase database;
  ShaderDatabaseTest test(database);
  EXPECT_FALSE(database.remove(nullptr));
  test_all_cases_remove<SHADERTYPE_LIST>(test);
}

template<class HEAD, class... TAIL>
void test_all_cases_get(const ShaderDatabaseTest &test) {
  Shader *ptr = test.database.get(HEAD::getType_static());
  EXPECT_NE(ptr, nullptr);
  if constexpr (sizeof...(TAIL) > 0)
    test_all_cases_contain<TAIL...>(test);
}

TEST(ShaderDatabaseTest, get) {
  ShaderDatabase database;
  ShaderDatabaseTest test(database);
  int size = database.size();
  test_all_cases_get<SHADERTYPE_LIST>(test);
  EXPECT_EQ(size, database.size());
}