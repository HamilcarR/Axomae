#include "../includes/SceneHierarchy.h"
#include "../includes/SceneNodeBuilder.h"
#include <time.h>
#include <gtest/gtest.h>

constexpr unsigned int TEST_TREE_MAX_DEPTH = 4; 
constexpr unsigned int TEST_TREE_MAX_NODE_DEGREE = 3 ;  
class SceneTreeBuilder{
public:
    SceneTreeBuilder(){
        srand(time(nullptr));
        node_count = 0 ; 
        leaf_count = 0 ; 
    }
    virtual ~SceneTreeBuilder(){}

    void buildSceneTree(unsigned int depth , unsigned int max_degree){
        SceneNodeInterface *root = SceneNodeBuilder::buildEmptyNode(nullptr);
        tree.setRoot(root);
        tree.addGenericNodeToDelete(root);  
        node_count ++ ; 
        buildRecursive(root , max_degree , depth);
        tree.updateOwner();
    }

    SceneTree* getTreePointer(){return &tree;}

    unsigned getLeafCount(){return leaf_count;}

    unsigned getNodeCount(){return node_count; }
    
    void clean(){tree.clear(); }
private:
    void buildRecursive(SceneNodeInterface* node , unsigned max_degree , unsigned depth){
        if(depth == 0){
            leaf_count ++; 
            return ; 
        }
        int size_numbers = rand() % max_degree + 1 ;
        for(int i = 0 ; i < size_numbers ; i++){
            SceneNodeInterface* child = SceneNodeBuilder::buildEmptyNode(node);
            tree.addGenericNodeToDelete(child);
            node_count ++ ; 
            buildRecursive(child , max_degree , depth - 1);  
        }
    }

protected:
    SceneTree tree ;
    unsigned node_count ; 
    unsigned leaf_count ; 
};


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
    static void testLeafCount(SceneNodeInterface* node , unsigned int *leaf_number){
        if(node->isLeaf())
            (*leaf_number)++;
    }
};


TEST(DFSTest , dfsNodeCount){
    SceneTreeBuilder builder; 
    builder.buildSceneTree(TEST_TREE_MAX_DEPTH , TEST_TREE_MAX_NODE_DEGREE); 
    SceneTree* tree = builder.getTreePointer(); 
    unsigned int i = 0 ;
    tree->dfs(tree->getRootNode() , &PseudoFunctors::testNodeCount , &i);
    builder.clean();
    EXPECT_EQ(i , builder.getNodeCount());

}

TEST(DFSTest , updateAccumulatedTransformations){
    SceneTreeBuilder builder;
    builder.buildSceneTree(TEST_TREE_MAX_DEPTH , TEST_TREE_MAX_NODE_DEGREE);  
    SceneTree *tree = builder.getTreePointer(); 
    auto root_local_transf = tree->getRootNode()->getLocalModelMatrix(); 
    auto updated_local_transf = glm::translate(root_local_transf , glm::vec3(1. , 0. , 0.));
    tree->getRootNode()->setLocalModelMatrix(updated_local_transf); 
    tree->updateAccumulatedTransformations();
    std::vector<glm::mat4> matrices; 
    tree->dfs(tree->getRootNode() , &PseudoFunctors::testTransformationPropagation , matrices);
    for(auto m : matrices){
        bool eq = m == updated_local_transf ; 
        ASSERT_EQ(eq , true);
    } 
    builder.clean(); 
}


TEST(DFSTest , leafCount){
    SceneTreeBuilder builder; 
    builder.buildSceneTree(TEST_TREE_MAX_DEPTH , TEST_TREE_MAX_NODE_DEGREE);
    SceneTree *tree = builder.getTreePointer();
    unsigned i = 0 ; 
    tree->dfs(tree->getRootNode() , &PseudoFunctors::testLeafCount , &i);
    EXPECT_EQ(i , builder.getLeafCount()); 

}