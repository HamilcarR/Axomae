#include "ArcballCamera.h"
#include "Camera.h"
#include "DatabaseBuilderTest.h"
#include "FreePerspectiveCamera.h"
#include "LightingSystem.h"
#include "Mesh.h"

#define NODETYPE_LIST \
  SceneTreeNode, Mesh, DirectionalLight, PointLight, SpotLight, ArcballCamera, FreePerspectiveCamera, BoundingBoxMesh, CubeMesh, QuadMesh

const int COUNT = 10;  // Number of subtypes
namespace node_database_test {
  template<class HEAD, class... TAIL>
  void addNode(IResourceDB<int, datastructure::NodeInterface> &database) {
    math::random::CPUPseudoRandomGenerator generator;
    bool persistence = generator.randb();
    database::node::store<HEAD>(database, persistence);
    if constexpr (sizeof...(TAIL) > 0)
      addNode<TAIL...>(database);
  }

};  // namespace node_database_test

class NodeDatabaseTest final : public DatabaseBuilderTest<int, datastructure::NodeInterface> {
 public:
  explicit NodeDatabaseTest(INodeDatabase &db) : DatabaseBuilderTest<int, datastructure::NodeInterface>(db) { buildDatabase(); }

  template<class TYPE, class... Args>
  database::Result<int, TYPE> addNode(INodeDatabase &database, bool persistence, Args &&...args) {
    database::Result<int, TYPE> result = database::node::store<TYPE>(database, persistence, std::forward<Args>(args)...);
    return result;
  }

 private:
  void buildDatabase() { node_database_test::addNode<NODETYPE_LIST>(database); }
};

TEST(NodeDatabaseTest, add) {
  INodeDatabase database;
  NodeDatabaseTest test(database);
  EXPECT_EQ(database.size(), COUNT);
  test.addNode<Mesh>(database, true);
  EXPECT_EQ(database.size(), COUNT + 1);
}

TEST(NodeDatabaseTest, contains) {
  INodeDatabase database;
  NodeDatabaseTest test(database);
  const auto &map = database.getConstData();
  for (const auto &elem : map) {
    EXPECT_TRUE(database.contains(elem.first));
  }
}

static void test_all_remove(NodeDatabaseTest &test) {
  for (int i = 0; i < test.getDatabaseSize(); i++) {
    const auto it = test.database.getConstData().find(i);
    datastructure::NodeInterface *ptr = it->second.get();
    EXPECT_TRUE(test.database.remove(i));
    EXPECT_FALSE(test.database.remove(i));
    EXPECT_FALSE(test.database.remove(ptr));
  }
}

TEST(NodeDatabaseTest, remove) {
  INodeDatabase database;
  NodeDatabaseTest test(database);
  EXPECT_FALSE(database.remove(-1));
  EXPECT_FALSE(database.remove(database.size()));
  EXPECT_FALSE(database.remove(nullptr));
  test_all_remove(test);
}

static void test_all_get(NodeDatabaseTest &test) {
  EXPECT_EQ(test.database.get(-1), nullptr);
  const auto &map = test.database.getConstData();
  for (const auto &elem : map) {
    EXPECT_EQ(test.database.get(elem.first), elem.second.get());
  }
}

TEST(NodeDatabaseTest, get) {
  INodeDatabase database;
  NodeDatabaseTest test(database);
  int size = database.size();
  database.get(0);
  EXPECT_EQ(size, database.size());
  test_all_get(test);
}

TEST(NodeDatabaseTest, firstFreeId) {
  INodeDatabase database;
  NodeDatabaseTest test(database);
  int ffid = database.firstFreeId();
  EXPECT_EQ(ffid, COUNT);
  database.remove(0);
  ffid = database.firstFreeId();
  EXPECT_EQ(ffid, 0);
}
