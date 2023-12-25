
#include "DatabaseBuilderTest.h"

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