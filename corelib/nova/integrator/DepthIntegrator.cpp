#include "DepthIntegrator.h"

namespace nova::integrator {

  static float normalize_depth(float z, float near, float far) { return (2.0f * near * far) / (far + near - z * (far - near)); }

  /* same as regular integrators , but sample pixels without random deviation and sample loop*/
  void DepthIntegrator::render(RenderBuffers<float> *buffers, Tile &tile, nova_eng_internals &nova_internals) const {
    const NovaResourceManager *nova_resource_manager = nova_internals.resource_manager;
    NovaExceptionManager *nova_exception_manager = nova_internals.exception_manager;

    const camera::CameraResourcesHolder &camera = nova_resource_manager->getCameraData();
    float near = camera.near;
    float far = camera.far;

    sampler::RandomSampler random_sampler = sampler::RandomSampler();
    sampler::SamplerInterface sampler = &random_sampler;
    const scene::SceneTransformations &scene_transformations = nova_resource_manager->getSceneTransformation();

    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {

        validate(sampler, nova_internals);
        if (nova_exception_manager->checkErrorStatus() != exception::NOERR) {
          prepareAbortRender();
          return;
        }

        unsigned int idx = generateImageOffset(tile, nova_resource_manager->getEngineData().vertical_invert, x, y);

        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        if (!nova_resource_manager->getEngineData().is_rendering)
          return;
        math::camera::camera_ray r = math::camera::ray_inv_mat(
            ndc.x, ndc.y, nova_resource_manager->getCameraData().inv_P, nova_resource_manager->getCameraData().inv_V);
        Ray ray(r.near, r.far);

        glm::vec4 distance = Li(ray, nova_internals, 0, sampler);
        float depth = 1 - (distance.x - near) / (far - near);
        depth = normalize_depth(depth, near, far) * 2.f - 1.f;
        glm::vec4 rgb{depth / far};
        if (distance.x == MAXFLOAT)
          rgb = glm::vec4(far);
        accumulateRgbRenderbuffer(buffers, idx, rgb);
      }
    tile.finished_render = true;
  }

  /* returns closest primitive distance (intersection) , and farthest primitive distance*/
  glm::vec4 DepthIntegrator::Li(const Ray &ray, nova_eng_internals &nova_internals, int /*depth*/, sampler::SamplerInterface & /*sampler*/) const {
    bvh_hit_data hit = bvh_hit(ray, nova_internals);
    if (hit.is_hit) {
      glm::vec3 min_max_intersect{hit.hit_d.t, hit.prim_max_t, MAXFLOAT};
      return {min_max_intersect, 1.f};
    }
    return glm::vec4(MAXFLOAT);
  }
}  // namespace nova::integrator