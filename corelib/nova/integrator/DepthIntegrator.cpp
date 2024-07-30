#include "DepthIntegrator.h"
#include "Mutex.h"

namespace nova::integrator {

  static float normalize_depth(float z, float near, float far) { return (2.0f * near * far) / (far + near - z * (far - near)); }

  /* same as regular integrators , but sample pixels without random deviation and sample loop*/
  void DepthIntegrator::render(RenderBuffers<float> *buffers, Tile &tile, const NovaResourceManager *nova_resource_manager) const {
    static std::mutex mutex;
    const camera::CameraResourcesHolder &camera = nova_resource_manager->getCameraData();
    float near = camera.getNear();
    float far = camera.getFar();

    const scene::SceneTransformations &scene_transformations = nova_resource_manager->getSceneTransformation();
    for (int y = tile.height_end - 1; y >= tile.height_start; y = y - 1)
      for (int x = tile.width_start; x < tile.width_end; x = x + 1) {
        unsigned int idx = 0;
        if (!nova_resource_manager->getEngineData().isAxisVInverted())
          idx = (y * tile.image_total_width + x) * 4;
        else
          idx = ((tile.image_total_height - 1 - y) * tile.image_total_width + x) * 4;

        const glm::vec2 ndc = math::camera::screen2ndc(x, tile.image_total_height - y, tile.image_total_width, tile.image_total_height);
        if (*nova_resource_manager->getEngineData().getCancelPtr())
          return;
        math::camera::camera_ray r = math::camera::ray_inv_mat(
            ndc.x, ndc.y, nova_resource_manager->getCameraData().getInvProjection(), nova_resource_manager->getCameraData().getInvView());
        Ray ray(r.near, r.far);
        sampler::HammersleySampler ham = sampler::HammersleySampler();
        sampler::SamplerInterface sampler = &ham;
        glm::vec4 distance = Li(ray, nova_resource_manager, 0, sampler);

        float depth = 1 - (distance.x - near) / (far - near);
        depth = normalize_depth(depth, near, far) * 2.f - 1.f;
        glm::vec3 rgb{depth / far};
        if (distance.x == MAXFLOAT)
          rgb = glm::vec3(far);
        for (int k = 0; k < 3; k++)
          buffers->accumulator_buffer[idx + k] += buffers->partial_buffer[idx + k];

        buffers->partial_buffer[idx] = rgb.r;
        buffers->partial_buffer[idx + 1] = rgb.g;
        buffers->partial_buffer[idx + 2] = rgb.b;
        buffers->partial_buffer[idx + 3] = 1.f;
        buffers->accumulator_buffer[idx + 3] = 1.f;
      }
    tile.finished_render = true;
  }

  /* returns closest primitive distance (intersection) , and farthest primitive distance*/
  glm::vec4 DepthIntegrator::Li(const Ray &ray,
                                const NovaResourceManager *nova_resources,
                                int /*depth*/,
                                sampler::SamplerInterface & /*sampler*/) const {
    bvh_hit_data hit = bvh_hit(ray, nova_resources);
    if (hit.is_hit) {
      glm::vec3 min_max_intersect{hit.hit_d.t, hit.prim_max_t, MAXFLOAT};
      return {min_max_intersect, 1.f};
    }
    return glm::vec4(MAXFLOAT);
  }
}  // namespace nova::integrator