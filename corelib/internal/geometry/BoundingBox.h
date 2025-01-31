#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <internal/common/axstd/span.h>
#include <internal/common/math/math_utils.h>
#include <internal/macro/project_macros.h>
#include <vector>

/**
 * @brief AABB
 * @file BoundingBox.h
 */

namespace geometry {
  class BoundingBox {

    ax_device_callable friend BoundingBox operator*(const glm::mat4 &matrix, const BoundingBox &bounding_box) {
      glm::vec3 min_c = matrix * glm::vec4(bounding_box.getMinCoords(), 1.f);
      glm::vec3 max_c = matrix * glm::vec4(bounding_box.getMaxCoords(), 1.f);
      return {min_c, max_c};
    }

   private:
    glm::vec3 max_coords{-1e32f};
    glm::vec3 min_coords{1e32f};
    glm::vec3 center{};

   public:
    CLASS_DCM(BoundingBox)

    ax_device_callable explicit BoundingBox(const float *geometry, size_t size);
    ax_device_callable BoundingBox(const glm::vec3 &min_coords, const glm::vec3 &max_coords);
    ax_device_callable BoundingBox(const float min_coords[3], const float max_coords[3]);
    ax_device_callable explicit BoundingBox(const axstd::span<float> &geometry);

    /* Generates the maximum bounding box of a bounding boxes collection*/
    explicit BoundingBox(const std::vector<BoundingBox> &bboxes);
    explicit BoundingBox(const std::vector<glm::vec3> &vertices);
    explicit BoundingBox(const std::vector<float> &geometry);
    ax_device_callable ax_no_discard const glm::vec3 &getPosition() const { return center; }
    /**
     * @brief Returns the index + vertices array representatives of the bounding box
     * @return std::pair<std::vector<float> , std::vector<unsigned>>
     */
    ax_no_discard std::pair<std::vector<float>, std::vector<unsigned>> getVertexArray() const;
    ax_device_callable ax_no_discard const glm::vec3 &getMaxCoords() const { return max_coords; }
    ax_device_callable ax_no_discard const glm::vec3 &getMinCoords() const { return min_coords; }
    ax_device_callable void setMaxCoords(glm::vec3 max) { max_coords = max; }
    ax_device_callable void setMinCoords(glm::vec3 min) { min_coords = min; }

    ax_device_callable ax_no_discard bool operator==(const BoundingBox &other) const;
    ax_device_callable ax_no_discard BoundingBox operator+(const BoundingBox &addbox) const;
    /**
     * @brief Takes normalized ray direction , in object space
     */
    ax_device_callable ax_no_discard bool intersect(const glm::vec3 &ray_dir, const glm::vec3 &origin) const;
    ax_device_callable ax_no_discard float intersect(const glm::vec3 &ray_dir, const glm::vec3 &origin, float dist_min) const;
    ax_device_callable ax_no_discard float area() const;
  };

  class AABBInterface {
   public:
    virtual ~AABBInterface() = default;
    ax_no_discard virtual BoundingBox computeAABB() const = 0;
  };

}  // namespace geometry
#endif
