#include "../includes/SceneHierarchy.h"
#include <gtest/gtest.h>




class PseudoFunctors{
public:
    static void testDummyFunction(SceneNodeInterface* node) {
        std::cout << node << "\n"; 
    }
    static void testNodeCount(SceneNodeInterface* node , unsigned int *i){
        (*i)++; 
    }
    static void testTransformationPropagation(SceneNodeInterface* node , std::vector<glm::mat4> &matrices){
        glm::mat4 m = node->computeFinalTransformation(); 
        matrices.push_back(m); 
    }
};


TEST(DFSTest , dfsNodeCount){
    SceneTree tree; 
    SceneTreeNode test_root ;
    SceneTreeNode child1 = SceneTreeNode(&test_root);
    SceneTreeNode child2 = SceneTreeNode(&test_root); 
    SceneTreeNode child3 = SceneTreeNode(&child1);
    SceneTreeNode child4 = SceneTreeNode(&child1);
    tree.setRoot(&test_root);
    unsigned int i = 0 ;
    tree.dfs(tree.getRootNode() , &PseudoFunctors::testNodeCount , &i);
    EXPECT_EQ(i , 5);

}

TEST(DFSTest , updateAccumulatedTransformations){
    SceneTree tree; 
    SceneTreeNode test_root ;
    auto root_local_transf = test_root.getLocalModelMatrix(); 
    auto updated_local_transf = glm::translate(root_local_transf , glm::vec3(1. , 0. , 0.));
    test_root.setLocalModelMatrix(updated_local_transf); 
    SceneTreeNode child1 = SceneTreeNode(&test_root);
    SceneTreeNode child2 = SceneTreeNode(&test_root); 
    SceneTreeNode child3 = SceneTreeNode(&child1);
    SceneTreeNode child4 = SceneTreeNode(&child1);
    tree.setRoot(&test_root);
    tree.updateAccumulatedTransformations();
    std::vector<glm::mat4> matrices; 
    tree.dfs(tree.getRootNode() , &PseudoFunctors::testTransformationPropagation , matrices);
    for(auto m : matrices){
        bool eq = m == updated_local_transf ; 
        ASSERT_EQ(eq , true);
    } 

}


