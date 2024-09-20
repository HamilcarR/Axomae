#ifndef HierarchyInterface_H
#define HierarchyInterface_H
#include "NodeInterface.h"
#include <string>
#include <vector>

namespace datastructure {
  /**
   *@file HierarchyInterface.h
   */

  class HierarchyInterface {
   public:
    virtual ~HierarchyInterface() = default;
    virtual void setRoot(NodeInterface *_root) = 0;
    [[nodiscard]] virtual NodeInterface *getIterator() const = 0;
    virtual void setIterator(NodeInterface *iter) = 0;
    [[nodiscard]] virtual NodeInterface *getRootNode() const = 0;
    virtual void setAsRootChild(NodeInterface *node) = 0;
    /**
     * @brief Update owner of each node to this structure
     */
    virtual void updateOwner() = 0;
    virtual void clear() = 0;
    virtual std::vector<NodeInterface *> findByName(const std::string &name) = 0;
  };

}  // namespace datastructure
#endif  // HierarchyInterface_H
