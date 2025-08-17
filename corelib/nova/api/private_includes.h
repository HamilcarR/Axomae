#ifndef PRIVATE_INCLUDES_H
#define PRIVATE_INCLUDES_H
#include "../api.h"
#include "api_common.h"
#include "api_scene.h"
#include "api_transform.h"
#include "camera/nova_camera.h"
#include "engine/datastructures.h"
#include "manager/NovaResourceManager.h"
#include "material/NovaMaterials.h"
#include "scene/nova_scene.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/math/utils_3D.h>
#include <internal/geometry/Object3D.h>

constexpr int PBR_TEXTURE_PACK_SIZE = 8;
namespace nova {

  class NvTransform final : public Transform {
    glm::mat4 transform;

   public:
    NvTransform();
    NvTransform(const Transform &transform);
    NvTransform &operator=(const Transform &transform);
    void setTransformMatrix(const float transform[16]) override;
    void rotateQuat(float x, float y, float z, float w) override;
    void rotateEuler(float angle, float x, float y, float z) override;
    void reset() override;
    void getTransformMatrix(float transform[16]) const override;
    void translate(float x, float y, float z) override;
    void scale(float scale) override;
    void scale(float x, float y, float z) override;

    void setTranslation(float x, float y, float z) override;
    void getTranslation(float translation[3]) const override;
    void setScale(float x, float y, float z) override;
    void getScale(float scale[3]) const override;
    void getRotation(float rotation[4]) const override;
    void setRotation(const float rotation[4]) override;
    void multiply(const Transform &other) override;
    void getInverse(float inverse[16]) const override;
    void transpose(float result[16]) const override;
    void transposeInverse(float result[9]) const override;
  };

  class NvTexture final : public Texture {
   public:
    enum DATATYPE { F_ARRAY, I_ARRAY, SINGLE_COL };

   private:
    union {
      const float *f_buffer;
      const uint32_t *ui_buffer;
      float color[4];
    } memory{};
    DATATYPE type{};
    unsigned width{0};
    unsigned height{9};
    unsigned channel{0};
    GLuint interop_id{0};
    bool invert_y{false};
    bool invert_x{false};

   public:
    NvTexture() = default;
    NvTexture(const Texture &other);
    NvTexture &operator=(const Texture &other);

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
    const T *getData() const;
  };

  class NvMaterial final : public Material {
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
    NvMaterial(const Material &other);
    NvMaterial &operator=(const Material &other);

    ERROR_STATE registerAlbedo(const Texture &texture) override;
    ERROR_STATE registerNormal(const Texture &texture) override;
    ERROR_STATE registerMetallic(const Texture &texture) override;
    ERROR_STATE registerEmissive(const Texture &texture) override;
    ERROR_STATE registerRoughness(const Texture &texture) override;
    ERROR_STATE registerOpacity(const Texture &texture) override;
    ERROR_STATE registerSpecular(const Texture &texture) override;
    ERROR_STATE registerAmbientOcclusion(const Texture &texture) override;
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

  class NvTriMesh final : public TriMesh {
    Object3D attributes;
    NvTransform transform;
    trimesh_vbo_interop_s vbo_interop{};

   public:
    NvTriMesh() = default;
    NvTriMesh(const TriMesh &other);
    NvTriMesh &operator=(const TriMesh &other);
    ERROR_STATE registerBufferVertices(float *vertices, size_t num) override;
    ERROR_STATE registerBufferNormals(float *normals, size_t num) override;
    ERROR_STATE registerBufferTangents(float *tangents, size_t num) override;
    ERROR_STATE registerBufferBitangents(float *bitangents, size_t num) override;
    ERROR_STATE registerBufferColors(float *colors, size_t num) override;
    ERROR_STATE registerBufferUVs(float *uv, size_t num) override;
    ERROR_STATE registerBufferIndices(unsigned *indices, size_t num) override;
    ERROR_STATE registerTransform(const Transform &transform) override;
    ERROR_STATE registerInteropVertices(uint32_t vbo_id) override;
    ERROR_STATE registerInteropNormals(uint32_t vbo_id) override;
    ERROR_STATE registerInteropTangents(uint32_t vbo_id) override;
    ERROR_STATE registerInteropBitangents(uint32_t vbo_id) override;
    ERROR_STATE registerInteropColors(uint32_t vbo_id) override;
    ERROR_STATE registerInteropUVs(uint32_t vbo_id) override;
    ERROR_STATE registerInteropIndices(uint32_t vbo_id) override;

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

    const Transform &getTransform() const override;
  };

  struct trimesh_object_s {
    std::unique_ptr<NvTriMesh> mesh_geometry;
    std::unique_ptr<NvMaterial> mesh_material;
  };

  class NvCamera final : public Camera {
    camera::CameraResourcesHolder camera;
  };

  class NvScene final : public Scene {
    std::vector<trimesh_object_s> trimesh_group;
    std::vector<NvTexture> envmaps;
    std::vector<NvCamera> cameras;

   public:
    ERROR_STATE addMesh(const TriMesh &mesh, const Material &material) override;
    ERROR_STATE addEnvmap(const Texture &envmap_texture) override;
    ERROR_STATE addCamera(const Camera &camera) override;
    ERROR_STATE addRootTransform(const float transform[16]) override;
    axstd::span<const trimesh_object_s> getTrimeshArray() const;
  };

  struct HostStoredRenderableBuffers {
    axstd::managed_vector<float> partial_buffer;
    axstd::managed_vector<float> accumulator_buffer;
    axstd::managed_vector<float> depth_buffer;
    axstd::managed_vector<float> normal_buffer;
    unsigned width{}, height{};
    static constexpr unsigned CHANNELS_COLOR = 4;
    static constexpr unsigned CHANNELS_DEPTH = 1;
    static constexpr unsigned CHANNELS_NORMALS = 3;
  };

  class NvRenderBuffer : public RenderBuffer {
    std::unique_ptr<HostStoredRenderableBuffers> render_buffers;

   public:
    ERROR_STATE createRenderBuffer(unsigned width, unsigned height) override;
    HdrBufferStruct getRenderBuffers() const override;
  };

  class NvEngineInstance : public Engine {
    NovaResourceManager manager;
    RenderBufferPtr render_buffer;
    bool uses_interops{false};
    bool uses_gpu{false};
    bool scene_built{false};

   public:
    ERROR_STATE buildScene(const Scene &scene) override;
    ERROR_STATE setRenderBuffers(RenderBufferPtr buffers) override;
    ERROR_STATE buildAcceleration() override;
    ERROR_STATE useInterops(bool use) override;
    ERROR_STATE useGpu(bool use) override;
    ERROR_STATE cleanup() override;
  };

  Object3D to_obj3d(const NvTriMesh &trimesh);
  material::NovaMaterialInterface setup_material_data(const AbstractMesh &mesh, const NvMaterial &material, NovaResourceManager &manager);
  void setup_geometry_data(const TriMesh &mesh,
                           const float final_transform[16],
                           material::NovaMaterialInterface &material,
                           NovaResourceManager &manager,
                           std::size_t mesh_index,
                           bool uses_interops);

}  // namespace nova
#endif
