add_library(ax_datastructure
        AbstractHierarchy.cpp
        AbstractNode.cpp

        btree/BNode.hpp
        btree/BTree.hpp
)


target_link_libraries(${PROJECT_NAME} PUBLIC ax_datastructure)
target_link_libraries(ax_datastructure)




target_include_directories(ax_datastructure
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

