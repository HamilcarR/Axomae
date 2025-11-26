#include "common_routines.cuh"

// TODO: Provide unified API between CPU integrator and GPU integrator.

template<typename... Args>
ax_device_force_inlined void AX_THROW(unsigned code, Args... detail) {
  optixThrowException(code, detail...);
}

ax_device_force_inlined nova::intersection_record_s payload2hitd(const path_payload_s &pld, const nova::Ray &wray) {
  nova::intersection_record_s hit_d;
  glm::vec3 normal = glm::vec3(pld.normal_matrix[0], pld.normal_matrix[1], pld.normal_matrix[2]);
  glm::vec3 tangent = glm::vec3(pld.normal_matrix[3], pld.normal_matrix[4], pld.normal_matrix[5]);
  glm::vec3 bitangent = glm::vec3(pld.normal_matrix[6], pld.normal_matrix[7], pld.normal_matrix[8]);
  hit_d.geometric_normal = normal;
  hit_d.binormal = tangent;
  hit_d.wo_dot_n = glm::dot(-wray.direction, normal);
  hit_d.u = pld.u;
  hit_d.v = pld.v;
  hit_d.t = pld.t;
  hit_d.position = wray.pointAt(pld.t);
  return hit_d;
}

extern "C" ax_kernel void __raygen__main() {
  AX_ASSERT_NOTNULL(parameters.handle);
  nova::device_traversal_param_s params = get_params();
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float u = (float)idx.x / (float)dim.x;
  float v = (float)idx.y / (float)dim.y;

  nv_sam::SobolSampler sobol_sampler_instance(ax_rqmc_engine);
  nv_sam::SamplerInterface sampler = &sobol_sampler_instance;
  register_stack_t<> reg{};
  glm::vec4 color{};
  StackAllocator allocator;
  sampler.reset((idx.x * dim.y + idx.y) * params.sample_max + params.sample_index);
  float rand[2];
  sampler.sample2D(rand);
  rand[0] *= 0.0005;
  rand[1] *= 0.0005;

  const glm::vec2 ndc = math::camera::uv2ndc(u, v);
  math::camera::camera_ray c_ray = math::camera::ray_inv_mat(ndc.x + rand[0], ndc.y + rand[1], params.camera.inv_P, params.camera.inv_V);
  float3 origin = {c_ray.near.x, c_ray.near.y, c_ray.near.z};
  float3 direction = {c_ray.far.x, c_ray.far.y, c_ray.far.z};
  glm::vec4 attenuation = glm::vec4(1.f);
  for (uint32_t depth = 0; depth < params.depth; depth++) {
    optixTrace(parameters.handle,
               origin,
               direction,
               0.001f,
               1e30f,
               0.f,
               OptixVisibilityMask(0xFF),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0,
               0,
               0,
               U32REG_VAR_LIST_15(reg));

    path_payload_s pld = unpack_payload(reg);
    nova::Ray wi = make_ray(origin, direction);

    if (pld.traversal_stopped) {  // AS has no hit, therefore sample envmap with current ray.
      attenuation *= sample_envmap(wi, &params);
      break;
    }
    nv_prim::NovaPrimitiveInterface prim = ax_primitive_view[pld.prim_idx];
    nova::intersection_record_s hit_d = payload2hitd(pld, wi);
    nv_tex::TextureCtx ctx(ax_texture_bundle);
    nv_tex::texture_data_aggregate_s tex_aggregate;
    tex_aggregate.texture_ctx = &ctx;
    nv_mat::shading_data_s shading;
    shading.texture_aggregate = &tex_aggregate;
    nova::Ray wo;
    material_record_s mat_record = {};
    if (prim.scatter(wi, wo, hit_d, mat_record, sampler, allocator, shading)) {

      nova::Ray out = nova::Ray::spawn(mat_record.lobe.wi, hit_d.geometric_normal, hit_d.position);
      origin = {out.origin.x, out.origin.y, out.origin.z};
      direction = {out.direction.x, out.direction.y, out.direction.z};
      nova::Spectrum color = mat_record.lobe.f * mat_record.lobe.costheta / mat_record.lobe.pdf;
      nova::Spectrum emissive = mat_record.emissive;

      glm::vec4 e = glm::vec4(emissive.toRgb(), 1.f);
      glm::vec4 c = glm::vec4(color.toRgb(), 1.f);
      attenuation = DENAN(e + c * attenuation);
    } else {
      attenuation = glm::vec4(0.f);
      break;
    }
  }
  color += attenuation;
  render_buffer_s rb = {};
  rb.render_target = params.render_buffers.partial_buffer;
  rb.width = params.width;
  rb.height = params.height;
  shade(rb, u, v, color);
}

extern "C" ax_kernel void __closesthit__minimum_intersect() {
  path_payload_s pld{};
  float2 uv = ax_tri_barycentrics;
  pld.t = ax_ray_tmax;
  pld.prim_idx = ax_prim_idx;
  pld.traversal_stopped = false;
  geometry::face_data_tri face = get_primitive_triangle(pld.prim_idx);
  transform4x4_t transform = get_primitive_transform(pld.prim_idx);
  float3 ori = ax_wray_ori;
  float3 dir = ax_wray_dir;
  nova::Ray wi = make_ray(ori, dir);
  geometry::interpolates_s intp = compute_lerps(face, transform, uv.x, uv.y, pld.t, wi);
  pld.u = intp.uv[0];
  pld.v = intp.uv[1];
  for (int i = 0; i < 3; i++) {
    pld.normal_matrix[i] = intp.n[i];
    pld.normal_matrix[i + 3] = intp.tan[i];
    pld.normal_matrix[i + 6] = intp.bit[i];
  }

  register_stack_t<> reg;
  pack_payload(pld, reg);
  U32REG_SET_PAYLOAD_15(reg);
}

extern "C" ax_kernel void __miss__sample_envmap() {
  path_payload_s pld{};
  pld.traversal_stopped = true;
  register_stack_t<> reg;
  pack_payload(pld, reg);
  U32REG_SET_PAYLOAD_15(reg);
}

extern "C" ax_kernel void __anyhit__random_intersect() {}

extern "C" ax_kernel void __exception__exception_handler() {
  unsigned int type = optixGetExceptionCode();
  printf("Optix device exception : code=%u\n", type);
}
