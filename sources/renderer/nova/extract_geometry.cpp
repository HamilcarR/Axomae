#include "Drawable.h"
#include "Mesh.h"
#include "bake.h"
#include "extract_scene_internal.h"
#include "nova/bake_render_data.h"
namespace nova_baker_utils {

  void setup_mesh(const drawable_original_transform &drawable, nova::Trimesh &mesh) {
    nova::TransformPtr transform = nova::create_transform();
    const glm::mat4 &drawable_transform = drawable.mesh_original_transformation;
    transform->setTransformMatrix(glm::value_ptr(drawable_transform));
    mesh.registerTransform(std::move(transform));

    const Object3D &geometry = drawable.mesh->getMeshPointer()->getGeometry();
    mesh.registerBufferVertices(geometry.vertices.data(), geometry.vertices.size());
    mesh.registerBufferNormals(geometry.normals.data(), geometry.normals.size());
    mesh.registerBufferTangents(geometry.tangents.data(), geometry.tangents.size());
    mesh.registerBufferBitangents(geometry.bitangents.data(), geometry.bitangents.size());
    mesh.registerBufferColors(geometry.colors.data(), geometry.colors.size());
    mesh.registerBufferUVs(geometry.uv.data(), geometry.uv.size());
    mesh.registerBufferIndices(geometry.indices.data(), geometry.indices.size());

    const PackedGLGeometryBuffer &gl_buffers = drawable.mesh->getMeshGLBuffers();
    mesh.registerInteropVertices(gl_buffers.getVertexBufferID().getID());
    mesh.registerInteropNormals(gl_buffers.getNormalBufferID().getID());
    mesh.registerInteropTangents(gl_buffers.getTangentxBufferID().getID());
    mesh.registerInteropColors(gl_buffers.getColorBufferID().getID());
    mesh.registerInteropUVs(gl_buffers.getUVBufferID().getID());
    mesh.registerInteropIndices(gl_buffers.getIndexBufferID().getID());
  }
}  // namespace nova_baker_utils
