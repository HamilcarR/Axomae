#include "api_common.h"
#include "glm/gtc/type_ptr.hpp"
#include "private_includes.h"
#include <typeinfo>
namespace nova {

  struct trimesh_vbo_interop_s {
    uint32_t vbo_vertices;
    uint32_t vbo_normals;
    uint32_t vbo_tangents;
    uint32_t vbo_bitangents;
    uint32_t vbo_colors;
    uint32_t vbo_uvs;
    uint32_t vbo_indices;
  };

  class NvTrimesh final : public Trimesh {
    Object3D attributes;
    TransformPtr transform;
    trimesh_vbo_interop_s vbo_interop{};

   public:
    NvTrimesh() { transform = create_transform(); }

    ERROR_STATE registerBufferVertices(float *vertices, size_t num) override {
      if (!vertices || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.vertices = axstd::span<float>(vertices, num);
      return SUCCESS;
    }

    ERROR_STATE registerBufferNormals(float *normals, size_t num) override {
      if (!normals || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.normals = axstd::span<float>(normals, num);
      return SUCCESS;
    }

    ERROR_STATE registerBufferTangents(float *tangents, size_t num) override {
      if (!tangents || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.tangents = axstd::span<float>(tangents, num);
      return SUCCESS;
    }

    ERROR_STATE registerBufferBitangents(float *bitangents, size_t num) override {
      if (!bitangents || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.bitangents = axstd::span<float>(bitangents, num);
      return SUCCESS;
    }

    ERROR_STATE registerBufferColors(float *colors, size_t num) override {
      if (!colors || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.colors = axstd::span<float>(colors, num);
      return SUCCESS;
    }

    ERROR_STATE registerBufferUVs(float *uv, size_t num) override {
      if (!uv || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.uv = axstd::span<float>(uv, num);
      return SUCCESS;
    }

    ERROR_STATE registerBufferIndices(unsigned *indices, size_t num) override {
      if (!indices || num == 0)
        return INVALID_BUFFER_STATE;
      attributes.indices = axstd::span<unsigned>(indices, num);
      return SUCCESS;
    }

    ERROR_STATE registerTransform(TransformPtr trsf) override {
      if (!trsf)
        return INVALID_ARGUMENT;
      transform = std::move(trsf);
      return SUCCESS;
    }

    ERROR_STATE registerInteropVertices(uint32_t vbo_id) override {
      vbo_interop.vbo_vertices = vbo_id;
      return SUCCESS;
    }

    ERROR_STATE registerInteropNormals(uint32_t vbo_id) override {
      vbo_interop.vbo_normals = vbo_id;
      return SUCCESS;
    }

    ERROR_STATE registerInteropTangents(uint32_t vbo_id) override {
      vbo_interop.vbo_tangents = vbo_id;
      return SUCCESS;
    }

    ERROR_STATE registerInteropBitangents(uint32_t vbo_id) override {
      vbo_interop.vbo_bitangents = vbo_id;
      return SUCCESS;
    }

    ERROR_STATE registerInteropColors(uint32_t vbo_id) override {
      vbo_interop.vbo_colors = vbo_id;
      return SUCCESS;
    }

    ERROR_STATE registerInteropUVs(uint32_t vbo_id) override {
      vbo_interop.vbo_uvs = vbo_id;
      return SUCCESS;
    }

    ERROR_STATE registerInteropIndices(uint32_t vbo_id) override {
      vbo_interop.vbo_indices = vbo_id;
      return SUCCESS;
    }

    const Transform &getTransform() const override { return *transform; }

    axstd::span<float> getVertices() const override { return attributes.vertices; }

    axstd::span<float> getNormals() const override { return attributes.normals; }

    axstd::span<float> getTangents() const override { return attributes.tangents; }

    axstd::span<float> getBitangents() const override { return attributes.bitangents; }

    axstd::span<float> getColors() const override { return attributes.colors; }

    axstd::span<float> getUVs() const override { return attributes.uv; }

    axstd::span<unsigned> getIndices() const override { return attributes.indices; }

    uint32_t getInteropVertices() const override { return vbo_interop.vbo_vertices; }

    uint32_t getInteropNormals() const override { return vbo_interop.vbo_normals; }

    uint32_t getInteropTangents() const override { return vbo_interop.vbo_tangents; }

    uint32_t getInteropBitangents() const override { return vbo_interop.vbo_bitangents; }

    uint32_t getInteropColors() const override { return vbo_interop.vbo_colors; }

    uint32_t getInteropUVs() const override { return vbo_interop.vbo_uvs; }

    uint32_t getInteropIndices() const override { return vbo_interop.vbo_indices; }
  };

  Object3D to_obj3d(const Trimesh &trimesh) {
    Object3D obj3d;
    obj3d.vertices = trimesh.getVertices();
    obj3d.indices = trimesh.getIndices();
    obj3d.normals = trimesh.getNormals();
    obj3d.tangents = trimesh.getTangents();
    obj3d.bitangents = trimesh.getBitangents();
    obj3d.colors = trimesh.getColors();
    obj3d.uv = trimesh.getUVs();
    return obj3d;
  }

  TrimeshPtr create_trimesh() { return std::make_unique<NvTrimesh>(); }
}  // namespace nova
