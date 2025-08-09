#ifndef PRIVATE_INCLUDES_H
#define PRIVATE_INCLUDES_H
#include "../api.h"
#include "api_common.h"
#include "api_scene.h"
#include <internal/geometry/Object3D.h>

namespace nova {
  class NvTexture : public NvAbstractTexture {
   public:
    enum DATATYPE { F_ARRAY, I_ARRAY, SINGLE_COL };

   private:
    union {
      const float *f_buffer;
      const uint32_t *ui_buffer;
      float color[4];
    } memory;
    DATATYPE type;
    unsigned width{0};
    unsigned height{9};
    unsigned channel{0};
    GLuint interop_id{0};
    bool invert_y{false};
    bool invert_x{false};

   public:
    NvTexture() = default;
    NvTexture(const NvAbstractTexture &other);
    NvTexture &operator=(const NvAbstractTexture &other);

    ERROR_STATE setData(const uint32_t *buffer) override;
    ERROR_STATE setData(const float *buffer) override;
    ERROR_STATE setWidth(unsigned w) override;
    ERROR_STATE setHeight(unsigned h) override;
    ERROR_STATE setChannels(unsigned c) override;
    ERROR_STATE setInteropID(GLuint texture_id) override;
    ERROR_STATE invertY() override;
    ERROR_STATE invertX() override;

    DATATYPE getDataType() const { return type; }
    unsigned getWidth() const { return width; }
    unsigned getHeight() const { return height; }
    unsigned getChannels() const { return channel; }
    GLuint getInteropID() const { return interop_id; }
    bool getInvertY() const { return invert_y; }
    bool getInvertX() const { return invert_x; }

    template<class T>
    const T *getData() const {
      return &memory;
    }
  };

  class NvMaterial : public NvAbstractMaterial {
    NvTexture albedo;
    NvTexture normal;
    NvTexture metallic;
    NvTexture emissive;
    NvTexture roughness;
    NvTexture opacity;
    NvTexture specular;
    NvTexture ao;
    float refract_coeff{1.0f};
    float reflect_fuzz{0.0f};

   public:
    NvMaterial() = default;
    NvMaterial(const NvAbstractMaterial &other);
    NvMaterial &operator=(const NvAbstractMaterial &other);

    ERROR_STATE registerAlbedo(const NvAbstractTexture &texture) override;
    ERROR_STATE registerNormal(const NvAbstractTexture &texture) override;
    ERROR_STATE registerMetallic(const NvAbstractTexture &texture) override;
    ERROR_STATE registerEmissive(const NvAbstractTexture &texture) override;
    ERROR_STATE registerRoughness(const NvAbstractTexture &texture) override;
    ERROR_STATE registerOpacity(const NvAbstractTexture &texture) override;
    ERROR_STATE registerSpecular(const NvAbstractTexture &texture) override;
    ERROR_STATE registerAmbientOcclusion(const NvAbstractTexture &texture) override;
    ERROR_STATE setRefractCoeff(float eta) override;
    ERROR_STATE setReflectFuzz(float fuzz) override;

    NvTexture getAlbedo() const { return albedo; }
    NvTexture getNormal() const { return normal; }
    NvTexture getMetallic() const { return metallic; }
    NvTexture getEmissive() const { return emissive; }
    NvTexture getRoughness() const { return roughness; }
    NvTexture getOpacity() const { return opacity; }
    NvTexture getSpecular() const { return specular; }
    NvTexture getAmbientOcclusion() const { return ao; }
    float getRefractCoeff() const { return refract_coeff; }
    float getReflectFuzz() const { return reflect_fuzz; }
  };

  struct trimesh_vbo_interop_s {
    uint32_t vbo_vertices;
    uint32_t vbo_normals;
    uint32_t vbo_tangents;
    uint32_t vbo_bitangents;
    uint32_t vbo_colors;
    uint32_t vbo_uvs;
    uint32_t vbo_indices;
  };

  class NvTriMesh : public NvAbstractTriMesh {
    Object3D attributes;
    float transform[16]{};
    trimesh_vbo_interop_s vbo_interop{};

   public:
    NvTriMesh() = default;
    NvTriMesh(const NvAbstractTriMesh &other);
    NvTriMesh &operator=(const NvAbstractTriMesh &other);
    ERROR_STATE registerBufferVertices(float *vertices, size_t num) override;
    ERROR_STATE registerBufferNormals(float *normals, size_t num) override;
    ERROR_STATE registerBufferTangents(float *tangents, size_t num) override;
    ERROR_STATE registerBufferBitangents(float *bitangents, size_t num) override;
    ERROR_STATE registerBufferColors(float *colors, size_t num) override;
    ERROR_STATE registerBufferUVs(float *uv, size_t num) override;
    ERROR_STATE registerBufferIndices(unsigned *indices, size_t num) override;
    ERROR_STATE registerTransform(const float transform[16]) override;

    ERROR_STATE registerInteropVertices(uint32_t vbo_id) override;
    ERROR_STATE registerInteropNormals(uint32_t vbo_id) override;
    ERROR_STATE registerInteropTangents(uint32_t vbo_id) override;
    ERROR_STATE registerInteropBitangents(uint32_t vbo_id) override;
    ERROR_STATE registerInteropColors(uint32_t vbo_id) override;
    ERROR_STATE registerInteropUVs(uint32_t vbo_id) override;
    ERROR_STATE registerInteropIndices(uint32_t vbo_id) override;
  };

  struct trimesh_object_s {
    std::unique_ptr<NvTriMesh> mesh_geometry;
    std::unique_ptr<NvMaterial> mesh_material;
  };

  class NvScene : public NvAbstractScene {
    std::vector<trimesh_object_s> trimesh_group;

   public:
    ERROR_STATE addMesh(const NvAbstractTriMesh &mesh, const NvAbstractMaterial &material) override;
  };
}  // namespace nova
#endif
