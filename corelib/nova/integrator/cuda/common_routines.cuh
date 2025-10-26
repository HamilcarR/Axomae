#ifndef COMMON_ROUTINES_CUH
#define COMMON_ROUTINES_CUH
#include "camera/nova_camera.h"
#include "device_internal.h"
#include "gpu/nova_gpu.h"
#include "gpu/optix_params.h"
#include "primitive/PrimitiveInterface.h"
#include "ray/Ray.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
#include <internal/common/math/math_camera.h>
#include <internal/common/math/math_texturing.h>
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/memory/Allocator.h>
#include <optix_device.h>

#define U32REG_VAR_LIST_0(register_array_varname) register_array_varname[0]
#define U32REG_VAR_LIST_1(register_array_varname) U32REG_VAR_LIST_0(register_array_varname), register_array_varname[1]
#define U32REG_VAR_LIST_2(register_array_varname) U32REG_VAR_LIST_1(register_array_varname), register_array_varname[2]
#define U32REG_VAR_LIST_3(register_array_varname) U32REG_VAR_LIST_2(register_array_varname), register_array_varname[3]
#define U32REG_VAR_LIST_4(register_array_varname) U32REG_VAR_LIST_3(register_array_varname), register_array_varname[4]
#define U32REG_VAR_LIST_5(register_array_varname) U32REG_VAR_LIST_4(register_array_varname), register_array_varname[5]
#define U32REG_VAR_LIST_6(register_array_varname) U32REG_VAR_LIST_5(register_array_varname), register_array_varname[6]
#define U32REG_VAR_LIST_7(register_array_varname) U32REG_VAR_LIST_6(register_array_varname), register_array_varname[7]
#define U32REG_VAR_LIST_8(register_array_varname) U32REG_VAR_LIST_7(register_array_varname), register_array_varname[8]
#define U32REG_VAR_LIST_9(register_array_varname) U32REG_VAR_LIST_8(register_array_varname), register_array_varname[9]
#define U32REG_VAR_LIST_10(register_array_varname) U32REG_VAR_LIST_9(register_array_varname), register_array_varname[10]
#define U32REG_VAR_LIST_11(register_array_varname) U32REG_VAR_LIST_10(register_array_varname), register_array_varname[11]
#define U32REG_VAR_LIST_12(register_array_varname) U32REG_VAR_LIST_11(register_array_varname), register_array_varname[12]
#define U32REG_VAR_LIST_13(register_array_varname) U32REG_VAR_LIST_12(register_array_varname), register_array_varname[13]
#define U32REG_VAR_LIST_14(register_array_varname) U32REG_VAR_LIST_13(register_array_varname), register_array_varname[14]
#define U32REG_VAR_LIST_15(register_array_varname) U32REG_VAR_LIST_14(register_array_varname), register_array_varname[15]
#define U32REG_VAR_LIST_16(register_array_varname) U32REG_VAR_LIST_15(register_array_varname), register_array_varname[16]
#define U32REG_VAR_LIST_17(register_array_varname) U32REG_VAR_LIST_16(register_array_varname), register_array_varname[17]
#define U32REG_VAR_LIST_18(register_array_varname) U32REG_VAR_LIST_17(register_array_varname), register_array_varname[18]
#define U32REG_VAR_LIST_19(register_array_varname) U32REG_VAR_LIST_18(register_array_varname), register_array_varname[19]
#define U32REG_VAR_LIST_20(register_array_varname) U32REG_VAR_LIST_19(register_array_varname), register_array_varname[20]
#define U32REG_VAR_LIST_21(register_array_varname) U32REG_VAR_LIST_20(register_array_varname), register_array_varname[21]
#define U32REG_VAR_LIST_22(register_array_varname) U32REG_VAR_LIST_21(register_array_varname), register_array_varname[22]
#define U32REG_VAR_LIST_23(register_array_varname) U32REG_VAR_LIST_22(register_array_varname), register_array_varname[23]
#define U32REG_VAR_LIST_24(register_array_varname) U32REG_VAR_LIST_23(register_array_varname), register_array_varname[24]
#define U32REG_VAR_LIST_25(register_array_varname) U32REG_VAR_LIST_24(register_array_varname), register_array_varname[25]
#define U32REG_VAR_LIST_26(register_array_varname) U32REG_VAR_LIST_25(register_array_varname), register_array_varname[26]
#define U32REG_VAR_LIST_27(register_array_varname) U32REG_VAR_LIST_26(register_array_varname), register_array_varname[27]
#define U32REG_VAR_LIST_28(register_array_varname) U32REG_VAR_LIST_27(register_array_varname), register_array_varname[28]
#define U32REG_VAR_LIST_29(register_array_varname) U32REG_VAR_LIST_28(register_array_varname), register_array_varname[29]
#define U32REG_VAR_LIST_30(register_array_varname) U32REG_VAR_LIST_29(register_array_varname), register_array_varname[30]
#define U32REG_VAR_LIST_31(register_array_varname) U32REG_VAR_LIST_30(register_array_varname), register_array_varname[31]

#define U32REG_SET_PAYLOAD_0(register_array_varname) optixSetPayload_0(register_array_varname[0])
#define U32REG_SET_PAYLOAD_1(register_array_varname) \
  U32REG_SET_PAYLOAD_0(register_array_varname); \
  optixSetPayload_1(register_array_varname[1])
#define U32REG_SET_PAYLOAD_2(register_array_varname) \
  U32REG_SET_PAYLOAD_1(register_array_varname); \
  optixSetPayload_2(register_array_varname[2])
#define U32REG_SET_PAYLOAD_3(register_array_varname) \
  U32REG_SET_PAYLOAD_2(register_array_varname); \
  optixSetPayload_3(register_array_varname[3])
#define U32REG_SET_PAYLOAD_4(register_array_varname) \
  U32REG_SET_PAYLOAD_3(register_array_varname); \
  optixSetPayload_4(register_array_varname[4])
#define U32REG_SET_PAYLOAD_5(register_array_varname) \
  U32REG_SET_PAYLOAD_4(register_array_varname); \
  optixSetPayload_5(register_array_varname[5])
#define U32REG_SET_PAYLOAD_6(register_array_varname) \
  U32REG_SET_PAYLOAD_5(register_array_varname); \
  optixSetPayload_6(register_array_varname[6])
#define U32REG_SET_PAYLOAD_7(register_array_varname) \
  U32REG_SET_PAYLOAD_6(register_array_varname); \
  optixSetPayload_7(register_array_varname[7])
#define U32REG_SET_PAYLOAD_8(register_array_varname) \
  U32REG_SET_PAYLOAD_7(register_array_varname); \
  optixSetPayload_8(register_array_varname[8])
#define U32REG_SET_PAYLOAD_9(register_array_varname) \
  U32REG_SET_PAYLOAD_8(register_array_varname); \
  optixSetPayload_9(register_array_varname[9])
#define U32REG_SET_PAYLOAD_10(register_array_varname) \
  U32REG_SET_PAYLOAD_9(register_array_varname); \
  optixSetPayload_10(register_array_varname[10])
#define U32REG_SET_PAYLOAD_11(register_array_varname) \
  U32REG_SET_PAYLOAD_10(register_array_varname); \
  optixSetPayload_11(register_array_varname[11])
#define U32REG_SET_PAYLOAD_12(register_array_varname) \
  U32REG_SET_PAYLOAD_11(register_array_varname); \
  optixSetPayload_12(register_array_varname[12])
#define U32REG_SET_PAYLOAD_13(register_array_varname) \
  U32REG_SET_PAYLOAD_12(register_array_varname); \
  optixSetPayload_13(register_array_varname[13])
#define U32REG_SET_PAYLOAD_14(register_array_varname) \
  U32REG_SET_PAYLOAD_13(register_array_varname); \
  optixSetPayload_14(register_array_varname[14])
#define U32REG_SET_PAYLOAD_15(register_array_varname) \
  U32REG_SET_PAYLOAD_14(register_array_varname); \
  optixSetPayload_15(register_array_varname[15])
#define U32REG_SET_PAYLOAD_16(register_array_varname) \
  U32REG_SET_PAYLOAD_15(register_array_varname); \
  optixSetPayload_16(register_array_varname[16])
#define U32REG_SET_PAYLOAD_17(register_array_varname) \
  U32REG_SET_PAYLOAD_16(register_array_varname); \
  optixSetPayload_17(register_array_varname[17])
#define U32REG_SET_PAYLOAD_18(register_array_varname) \
  U32REG_SET_PAYLOAD_17(register_array_varname); \
  optixSetPayload_18(register_array_varname[18])
#define U32REG_SET_PAYLOAD_19(register_array_varname) \
  U32REG_SET_PAYLOAD_18(register_array_varname); \
  optixSetPayload_19(register_array_varname[19])
#define U32REG_SET_PAYLOAD_20(register_array_varname) \
  U32REG_SET_PAYLOAD_19(register_array_varname); \
  optixSetPayload_20(register_array_varname[20])
#define U32REG_SET_PAYLOAD_21(register_array_varname) \
  U32REG_SET_PAYLOAD_20(register_array_varname); \
  optixSetPayload_21(register_array_varname[21])
#define U32REG_SET_PAYLOAD_22(register_array_varname) \
  U32REG_SET_PAYLOAD_21(register_array_varname); \
  optixSetPayload_22(register_array_varname[22])
#define U32REG_SET_PAYLOAD_23(register_array_varname) \
  U32REG_SET_PAYLOAD_22(register_array_varname); \
  optixSetPayload_23(register_array_varname[23])
#define U32REG_SET_PAYLOAD_24(register_array_varname) \
  U32REG_SET_PAYLOAD_23(register_array_varname); \
  optixSetPayload_24(register_array_varname[24])
#define U32REG_SET_PAYLOAD_25(register_array_varname) \
  U32REG_SET_PAYLOAD_24(register_array_varname); \
  optixSetPayload_25(register_array_varname[25])
#define U32REG_SET_PAYLOAD_26(register_array_varname) \
  U32REG_SET_PAYLOAD_25(register_array_varname); \
  optixSetPayload_26(register_array_varname[26])
#define U32REG_SET_PAYLOAD_27(register_array_varname) \
  U32REG_SET_PAYLOAD_26(register_array_varname); \
  optixSetPayload_27(register_array_varname[27])
#define U32REG_SET_PAYLOAD_28(register_array_varname) \
  U32REG_SET_PAYLOAD_27(register_array_varname); \
  optixSetPayload_28(register_array_varname[28])
#define U32REG_SET_PAYLOAD_29(register_array_varname) \
  U32REG_SET_PAYLOAD_28(register_array_varname); \
  optixSetPayload_29(register_array_varname[29])
#define U32REG_SET_PAYLOAD_30(register_array_varname) \
  U32REG_SET_PAYLOAD_29(register_array_varname); \
  optixSetPayload_30(register_array_varname[30])
#define U32REG_SET_PAYLOAD_31(register_array_varname) \
  U32REG_SET_PAYLOAD_30(register_array_varname); \
  optixSetPayload_31(register_array_varname[31])

#define ax_index optixGetLaunchIndex()
#define ax_dim optixGetLaunchDimensions()
#define ax_wray_dir optixGetWorldRayDirection()
#define ax_wray_ori optixGetWorldRayOrigin()
#define ax_oray_ori optixGetObjectRayOrigin()
#define ax_oray_dir optixGetObjectRayDirection()
#define ax_ray_tmax optixGetRayTmax()
#define ax_terminate_ray optixTerminateRay()
#define ax_ignore_intersect optixIgnoreIntersection()
#define ax_prim_idx optixGetPrimitiveIndex()
#define ax_tri_barycentrics optixGetTriangleBarycentrics()
#define ax_rqmc_engine get_params().device_random_generators.rqmc_generator
#define ax_mesh_bundle get_params().mesh_bundle_views
#define ax_primitive_view get_params().primitives_view
#define ax_texture_bundle get_params().texture_bundle_views
#define ax_sbt_data_ptr(type) (type *)optixGetSbtDataPointer()
#define ax_normal2world(normal) optixTransformNormalFromObjectToWorldSpace(normal)

namespace nv_mat = nova::material;
namespace nv_shape = nova::shape;
namespace nv_prim = nova::primitive;
namespace nv_tex = nova::texturing;
namespace nv_sam = nova::sampler;

extern "C" {
extern ax_device_const nova::optix_traversal_param_s parameters;
}

ax_device_force_inlined const nova::device_traversal_param_s &get_params() { return parameters.d_params; }

ax_device_force_inlined nova::Ray make_ray(float3 ori, float3 dir) {
  return nova::Ray(glm::vec3(ori.x, ori.y, ori.z), glm::vec3(dir.x, dir.y, dir.z));
}

// TODO: Share functions with CPU integrators
ax_device_force_inlined void shade(render_buffer_s &render_buffer, float u, float v, const glm::vec4 &color, unsigned channels = 4) {
  unsigned px = math::texture::uvToPixel(u, render_buffer.width - 1);
  unsigned py = math::texture::uvToPixel(v, render_buffer.height - 1);
  unsigned offset = (py * render_buffer.width + px) * channels;
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}
ax_device_force_inlined void shade(render_buffer_s &render_buffer, unsigned x, unsigned y, const glm::vec4 &color, unsigned channels = 4) {
  unsigned offset = (y * render_buffer.width + x) * channels;
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}

ax_device_force_inlined void shade(render_buffer_s &render_buffer, unsigned offset, const glm::vec4 &color) {
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}

ax_device_force_inlined nova::Ray compute_screen_ray(float ndc_x, float ndc_y, const nova::camera::CameraResourcesHolder &camera) {
  math::camera::camera_ray c_ray = math::camera::ray_inv_mat(ndc_x, ndc_y, camera.inv_P, camera.inv_V);
  return {c_ray.near, c_ray.far};
}

ax_device_force_inlined glm::vec4 sample_envmap(const nova::Ray &ray, const nova::device_traversal_param_s *params, bool use_interop = true) {
  nova::texturing::texture_data_aggregate_s data = {};
  nova::texturing::TextureCtx texture_ctx(params->texture_bundle_views, use_interop);
  data.geometric_data.sampling_vector = ray.direction;
  data.texture_ctx = &texture_ctx;
  nova::texturing::EnvmapTexture img(params->current_envmap_index);
  return img.sample(0, 0, data);
}

ax_device_force_inlined void pack_payload(const path_payload_s &payload, register_stack_t<> reg) {
  reg[0] = payload.prim_idx;
  reg[1] = payload.prim_idx >> 32;
  reg[2] = *reinterpret_cast<const uint32_t *>(&payload.t);
  reg[3] = *reinterpret_cast<const uint32_t *>(&payload.u);
  reg[4] = *reinterpret_cast<const uint32_t *>(&payload.v);
  reg[5] = payload.traversal_stopped;

  // Object hit normal + tangent + bitangent.
  for (int i = 0; i < 9; i++)
    reg[6 + i] = *reinterpret_cast<const uint32_t *>(&payload.normal_matrix[i]);
}

ax_device_force_inlined path_payload_s unpack_payload(const register_stack_t<> reg) {
  path_payload_s payload;
  payload.prim_idx = ((uint64_t)(reg[1]) << 32 | reg[0]);
  payload.t = *reinterpret_cast<const float *>(&reg[2]);
  payload.u = *reinterpret_cast<const float *>(&reg[3]);
  payload.v = *reinterpret_cast<const float *>(&reg[4]);
  payload.traversal_stopped = reg[5];

  // Object hit normal + tangent + bitangent.
  for (int i = 0; i < 9; i++)
    payload.normal_matrix[i] = *reinterpret_cast<const float *>(&reg[6 + i]);

  return payload;
}

ax_device_force_inlined geometry::face_data_tri get_primitive_triangle(uint64_t prim_idx) {
  const nv_prim::NovaPrimitiveInterface &prim = ax_primitive_view[prim_idx];
  nv_shape::MeshCtx ctx(ax_mesh_bundle);
  nv_shape::face_data_s face_data = prim.getFace(ctx);
  AX_ASSERT_EQ(face_data.type, nv_shape::TRIANGLE);
  return face_data.data.triangle_face;
}

ax_device_force_inlined transform4x4_t get_primitive_transform(uint64_t prim_idx) {
  const nv_prim::NovaPrimitiveInterface &prim = ax_primitive_view[prim_idx];
  nv_shape::MeshCtx ctx(ax_mesh_bundle);
  return prim.getTransform(ctx);
}

ax_device_callable_inlined void fill_normals(const float3 normals[3],
                                             const float3 tangents[3],
                                             const float3 bitangents[3],
                                             geometry::face_data_tri &face) {
  face.n0[0] = normals[0].x;
  face.n0[1] = normals[0].y;
  face.n0[2] = normals[0].z;

  face.n1[0] = normals[1].x;
  face.n1[1] = normals[1].y;
  face.n1[2] = normals[1].z;

  face.n2[0] = normals[2].x;
  face.n2[1] = normals[2].y;
  face.n2[2] = normals[2].z;

  face.tan0[0] = tangents[0].x;
  face.tan0[1] = tangents[0].y;
  face.tan0[2] = tangents[0].z;

  face.tan1[0] = tangents[1].x;
  face.tan1[1] = tangents[1].y;
  face.tan1[2] = tangents[1].z;

  face.tan2[0] = tangents[2].x;
  face.tan2[1] = tangents[2].y;
  face.tan2[2] = tangents[2].z;

  face.bit0[0] = bitangents[0].x;
  face.bit0[1] = bitangents[0].y;
  face.bit0[2] = bitangents[0].z;

  face.bit1[0] = bitangents[1].x;
  face.bit1[1] = bitangents[1].y;
  face.bit1[2] = bitangents[1].z;

  face.bit2[0] = bitangents[2].x;
  face.bit2[1] = bitangents[2].y;
  face.bit2[2] = bitangents[2].z;
}

ax_device_callable_inlined geometry::interpolates_s compute_lerps(
    const geometry::face_data_tri &face, const transform4x4_t &transform, float u, float v, float t, const nova::Ray &ray) {
  float3 obj_normals[3] = {
      {face.n0[0], face.n0[1], face.n0[2]},
      {face.n1[0], face.n1[1], face.n1[2]},
      {face.n2[0], face.n2[1], face.n2[2]},
  };

  float3 obj_tangents[3] = {
      {face.tan0[0], face.tan0[1], face.tan0[2]},
      {face.tan1[0], face.tan1[1], face.tan1[2]},
      {face.tan2[0], face.tan2[1], face.tan2[2]},
  };

  float3 obj_bitangents[3] = {
      {face.bit0[0], face.bit0[1], face.bit0[2]},
      {face.bit1[0], face.bit1[1], face.bit1[2]},
      {face.bit2[0], face.bit2[1], face.bit2[2]},
  };

  // position is the transformed intersection point. Better to use it rather than using the internal v0 , v1 , v2 of face_data_tri as they aren't
  // interpolated.
  glm::vec3 position = ray.pointAt(t);

  geometry::face_data_tri transformed_face = {};
  face.copy(transformed_face);
  fill_normals(obj_normals, obj_tangents, obj_bitangents, transformed_face);
  geometry::interpolates_s interpolated = transformed_face.lerp(u, v);

  glm::vec3 w_normals = glm::normalize(transform.n * glm::vec3(interpolated.n[0], interpolated.n[1], interpolated.n[2]));
  glm::vec3 w_tangents = glm::normalize(transform.n * glm::vec3(interpolated.tan[0], interpolated.tan[1], interpolated.tan[2]));
  glm::vec3 w_bitangents = glm::normalize(transform.n * glm::vec3(interpolated.bit[0], interpolated.bit[1], interpolated.bit[2]));
  for (int i = 0; i < 3; i++) {
    interpolated.n[i] = ((float *)&w_normals)[i];
    interpolated.tan[i] = ((float *)&w_tangents)[i];
    interpolated.bit[i] = ((float *)&w_bitangents)[i];
  }

  interpolated.v[0] = position.x;
  interpolated.v[1] = position.y;
  interpolated.v[2] = position.z;
  return interpolated;
}

#endif
