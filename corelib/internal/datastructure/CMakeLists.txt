add_library(ax_core_datastructure
        AbstractHierarchy.cpp
        AbstractNode.cpp
        btree/BNode.hpp
        btree/BTree.hpp
)

target_link_libraries(ax_core PUBLIC ax_core_datastructure)
target_link_libraries(ax_core_datastructure
PRIVATE
        ax_core_common
)

target_include_directories(ax_core_datastructure
PUBLIC
    ${AXOMAE_CORELIB_ROOT}
)

