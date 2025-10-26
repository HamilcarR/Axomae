
#include "Integrator.h"
#include "DepthIntegrator.h"
#include "EmissiveIntegrator.h"
#include "NormalIntegrator.h"
#include "aggregate/acceleration_interface.h"
#include "engine/nova_engine.h"
#include "manager/ManagerInternalStructs.h"
#include "material/BxDF_math.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "spectrum/Spectrum.h"

namespace nova::integrator {

  void integrator_dispatch(RenderBuffers<float> *buffers, Tile &tile, nova::nova_eng_internals &nova_internals) {
    const engine::EngineResourcesHolder &engine = nova_internals.resource_manager->getEngineData();
    int integrator_type = engine.integrator_flag;
    if (integrator_type & PATH) {
      bool processed = false;
      if (integrator_type & COMBINED) {
        processed = true;
        const PathIntegrator path_integrator;
        path_integrator.render(buffers, tile, nova_internals);
      }
      if (integrator_type & NORMAL) {
        processed = true;
        const NormalIntegrator normal_integrator;
        normal_integrator.render(buffers, tile, nova_internals);
      }
      if (integrator_type & DEPTH) {
        processed = true;
        const DepthIntegrator depth_integrator;
        depth_integrator.render(buffers, tile, nova_internals);
      }
      if (integrator_type & EMISSIVE) {
        processed = true;
        const EmissiveIntegrator emissive_integrator;
        emissive_integrator.render(buffers, tile, nova_internals);
      }
      if (!processed)
        nova_internals.exception_manager->addError(nova::exception::INVALID_RENDER_MODE);
    } else {
      nova_internals.exception_manager->addError(nova::exception::INVALID_INTEGRATOR);
    }
  }

  bvh_hit_data bvh_hit(const Ray &ray, nova_eng_internals &nova_internals) {
    const NovaResourceManager *nova_resources = nova_internals.resource_manager;
    bvh_hit_data hit_ret{};
    hit_ret.last_primit = nullptr;
    hit_ret.is_rendering = &nova_internals.resource_manager->getEngineData().is_rendering;
    const aggregate::DefaultAccelerator &accel = nova_resources->getCpuManagedAccelerator();
    accel.hit(ray, hit_ret);
    return hit_ret;
  }

  glm::vec4 PathIntegrator::Li(const Ray &ray,
                               nova_eng_internals &nova_internals,
                               int depth,
                               sampler::SamplerInterface &sampler,
                               axstd::StaticAllocator64kb &allocator) const {
    const NovaResourceManager *nova_resources = nova_internals.resource_manager;
    bvh_hit_data hit = bvh_hit(ray, nova_internals);
    texturing::TextureCtx texture_context = texturing::TextureCtx(nova_internals.resource_manager->getTexturesData().getTextureBundleViews());
    texturing::texture_data_aggregate_s texture_sampling_data{};
    texture_sampling_data.texture_ctx = &texture_context;
    if (hit.is_hit) {
      Ray out{};
      material::shading_data_s shading{};
      shading.texture_aggregate = &texture_sampling_data;
      material_record_s mat_record{};
      if (depth < 0 || !hit.last_primit || !hit.last_primit->scatter(ray, out, hit.hit_d, mat_record, sampler, allocator, shading))
        return glm::vec4(0.f);

      Spectrum color = mat_record.lobe.f * mat_record.lobe.costheta / mat_record.lobe.pdf;
      Spectrum emit = mat_record.emissive;

      out = Ray::spawn(mat_record.lobe.wi, hit.hit_d.shading_frame.getNormal(), hit.hit_d.position);
      glm::vec4 next = Li(out, nova_internals, depth - 1, sampler, allocator);
      /* Here in case the value returned by the subsequent call to Li() is a NaN or Inf, we invalidate the color of the pixel altogether and set it
      * to zero. This helps keep a more uniform and precise value as the next sampling pass will be joined to the current pass and set a valid value
       to the pixel.*/
      glm::vec4 e = glm::vec4(emit.toRgb(), 1.f);
      glm::vec4 c = glm::vec4(color.toRgb(), 1.f);
      return DENAN(e + c * next);
    }
    glm::vec3 sample_vector = ray.direction;
    const auto &envmap = nova_resources->getTexturesData().getCurrentEnvmap();
    texture_sampling_data.geometric_data.sampling_vector = sample_vector;
    return envmap.sample(0, 0, texture_sampling_data);
  }

}  // namespace nova::integrator
