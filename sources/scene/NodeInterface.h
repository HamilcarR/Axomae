#ifndef NodeInterface_H
#define NodeInterface_H
#include <cstdlib>
#include <string>
#include <vector>
class AbstractHierarchy;

/**
 * @class NodeInterface
 * @brief Generic graph node. Can be used for any graph-like structures and processing , including scene manipulations
 */
class NodeInterface {

 public:
  virtual ~NodeInterface() = default;
  [[nodiscard]] virtual const std::vector<NodeInterface *> &getChildren() const = 0;
  [[nodiscard]] virtual const std::vector<NodeInterface *> &getParents() const = 0;
  virtual void addChildNode(NodeInterface *node) = 0;
  virtual void setParents(std::vector<NodeInterface *> &parents) = 0;
  virtual void emptyParents() = 0;
  virtual void emptyChildren() = 0;
  [[nodiscard]] virtual bool isLeaf() const = 0;
  virtual void setMark(bool marked) = 0;
  [[nodiscard]] virtual bool isMarked() const = 0;
  virtual void setName(const std::string &str) = 0;
  [[nodiscard]] virtual const std::string &getName() const = 0;
  virtual void setHierarchyOwner(AbstractHierarchy *_owner) = 0;
  [[nodiscard]] virtual AbstractHierarchy *getHierarchyOwner() const = 0;
  [[nodiscard]] virtual bool isRoot() const = 0;
  virtual NodeInterface *returnRoot() = 0;
  [[nodiscard]] virtual bool isUpdated() const = 0;
  virtual void reset() = 0;
};
#endif  // NodeInterface_H
