
#include "Integrator.h"
#include "DepthIntegrator.h"
#include "Logger.h"
#include "NormalIntegrator.h"
#include "engine/nova_engine.h"
#include "manager/ManagerInternalStructs.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"

namespace nova::integrator {

  void integrator_dispatch(RenderBuffers<float> *buffers, Tile &tile, nova::nova_eng_internals &nova_internals) {
    const engine::EngineResourcesHolder &engine = nova_internals.resource_manager->getEngineData();
    int integrator_type = engine.getIntegratorType();
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
      if (!processed)
        nova_internals.exception_manager->addError(nova::exception::INVALID_RENDER_MODE);
    } else {
      nova_internals.exception_manager->addError(nova::exception::INVALID_INTEGRATOR);
    }
  }

  bvh_hit_data bvh_hit(const Ray &ray, nova_eng_internals &nova_internals) {
    const NovaResourceManager *nova_resources = nova_internals.resource_manager;
    bvh_hit_data hit_ret;
    const primitive::NovaPrimitiveInterface *last_primit = nullptr;
    const aggregate::Bvhtl &bvh = nova_resources->getAccelerationData().accelerator;
    aggregate::bvh_helper_struct bvh_hit{MAXFLOAT, nullptr, &nova_resources->getEngineData().isRendering()};
    aggregate::base_options_bvh opts;
    opts.data = bvh_hit;
    hit_ret.is_hit = bvh.hit(ray, 0.0001f, MAXFLOAT, hit_ret.hit_d, &opts);
    hit_ret.last_primit = opts.data.last_prim;
    hit_ret.prim_min_t = opts.data.tmin;
    return hit_ret;
  }

  glm::vec4 PathIntegrator::Li(const Ray &ray, nova_eng_internals &nova_internals, int depth, sampler::SamplerInterface &sampler) const {
    const NovaResourceManager *nova_resources = nova_internals.resource_manager;
    bvh_hit_data hit = bvh_hit(ray, nova_internals);
    if (hit.is_hit) {
      Ray out{};
      if (!hit.last_primit || !hit.last_primit->scatter(ray, out, hit.hit_d, sampler) || depth < 0)
        return glm::vec4(0.f);
      glm::vec4 color = hit.hit_d.attenuation;
      glm::vec4 emit = hit.hit_d.emissive;
      return 10.f * emit + color * Li(out, nova_internals, depth - 1, sampler);
    }
    glm::vec3 sample_vector = ray.direction;
    return {texturing::sample_cubemap(sample_vector, &nova_resources->getEnvmapData()), 1.f};
  }

}  // namespace nova::integrator