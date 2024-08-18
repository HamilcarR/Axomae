#ifndef MeshInterface_H
#define MeshInterface_H
class Object3D;
class MeshInterface {
 public:
  virtual ~MeshInterface() = default;
  virtual void preRenderSetup() = 0;
  virtual void afterRenderSetup() = 0;
  [[nodiscard]] virtual bool isInitialized() const = 0;
  [[nodiscard]] virtual MaterialInterface *getMaterial() const = 0;
  [[nodiscard]] virtual const std::string &getMeshName() const = 0;
  virtual void setMeshName(const std::string &new_name) = 0;
  [[nodiscard]] virtual const Object3D &getGeometry() const = 0;
  virtual void setGeometry(const Object3D &geometry) = 0;
  virtual void setDrawState(bool draw) = 0;
  [[nodiscard]] virtual bool isDrawn() const = 0;
};

#endif  // MeshInterface_H
