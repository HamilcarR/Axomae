
#include "Camera.h"
#include "DatabaseBuilderTest.h"
#include "LightingSystem.h"
#include "Mesh.h"
namespace node_test {

  static void fill(DatabaseBuilderTest<int, INode>::ResultList &list, DatabaseBuilderTest<int, INode> &builder, bool persistence, int &stored) {
    list.push_back(builder.addNode<SceneTreeNode>(persistence));
    list.push_back(builder.addNode<Mesh>(persistence));
    list.push_back(builder.addNode<ArcballCamera>(persistence));
    list.push_back(builder.addNode<FreePerspectiveCamera>(persistence));
    list.push_back(builder.addNode<PointLight>(persistence));
    list.push_back(builder.addNode<CubeMesh>(persistence));
    list.push_back(builder.addNode<CubeMapMesh>(persistence));
    list.push_back(builder.addNode<QuadMesh>(persistence));
    list.push_back(builder.addNode<DirectionalLight>(persistence));
    list.push_back(builder.addNode<SpotLight>(persistence));
    list.push_back(builder.addNode<FrameBufferMesh>(persistence));
    list.push_back(builder.addNode<BoundingBoxMesh>(persistence));
    stored = list.size();  // Increment when adding new texture
  }

};  // namespace node_test

TEST(NodeDatabaseTest, add) {
  INodeDatabase database;
  DatabaseBuilderTest<int, INode> builder(database);
  DatabaseBuilderTest<int, INode>::ResultList list;
  int size;
  node_test::fill(list, builder, false, size);
  EXPECT_EQ(builder.stored.size(), size);
  for (auto &stored : builder.stored)
    EXPECT_GE(stored.first, 0);
  list.clear();
  INodeDatabase database2;
  DatabaseBuilderTest<int, INode> builder2(database2);
  node_test::fill(list, builder2, true, size);
  for (auto &stored : builder2.stored) {
    EXPECT_LE(stored.first, -1);
  }
}

TEST(NodeDatabaseTest, contains) {
  INodeDatabase database;
  DatabaseBuilderTest<int, INode> builder(database);
  DatabaseBuilderTest<int, INode>::ResultList list;
  int size;
  node_test::fill(list, builder, false, size);
  for (auto &stored : builder.stored) {
    database::Result<int, INode> result = builder.database.contains(stored.second.get());
    EXPECT_EQ(result.object, stored.second.get());
    EXPECT_EQ(result.id, stored.first);
  }
  database::Result<int, INode> result = builder.database.contains(nullptr);
  EXPECT_EQ(result.object, nullptr);
}

TEST(NodeDatabaseTest, remove) {
  INodeDatabase database;
  DatabaseBuilderTest<int, INode> builder(database);
  DatabaseBuilderTest<int, INode>::ResultList list;
  int size;
  node_test::fill(list, builder, false, size);
  bool test = builder.database.remove(list.size() + 1);
  EXPECT_FALSE(test);
  test = builder.database.remove(nullptr);
  EXPECT_FALSE(test);
  for (auto elem : list)
    EXPECT_TRUE(builder.database.remove(elem.id));
}

TEST(NodeDatabaseTest, get) {
  INodeDatabase database;
  DatabaseBuilderTest<int, INode> builder(database);
  DatabaseBuilderTest<int, INode>::ResultList list;
  int size;
  node_test::fill(list, builder, false, size);
  for (auto elem : list) {
    INode *tex = builder.database.get(elem.id);
    EXPECT_EQ(tex, builder.stored.at(elem.id).get());
  }
}
