#ifndef NOVA_CAMERA_H
#define NOVA_CAMERA_H
#include "math_utils.h"
#include "project_macros.h"
namespace nova::camera {
  class CameraResourcesHolder {
   private:
    unsigned int screen_width;
    unsigned int screen_height;
    /*Projection*/
    glm::mat4 P;
    glm::mat4 inv_P;
    /*View*/
    glm::mat4 V;
    glm::mat4 inv_V;
    /* Projection * View*/
    glm::mat4 PV;
    glm::mat4 inv_PV;
    glm::vec3 position;
    glm::vec3 up_vector;
    glm::vec3 direction;

   public:
    CLASS_CM(CameraResourcesHolder)

    void setPosition(const glm::vec3 &position);
    void setUpVector(const glm::vec3 &up);
    void setProjection(const glm::mat4 &projection);
    void setInvProjection(const glm::mat4 &inv_projection);
    void setView(const glm::mat4 &view);
    void setInvView(const glm::mat4 &view);
    void setDirection(const glm::vec3 &direction);
    void setScreenWidth(int width);
    void setScreenHeight(int height);
    [[nodiscard]] const glm::vec3 &getUpVector() const;
    [[nodiscard]] const glm::mat4 &getProjection() const;
    [[nodiscard]] const glm::mat4 &getInvProjection() const;
    [[nodiscard]] const glm::mat4 &getView() const;
    [[nodiscard]] const glm::mat4 &getInvView() const;
    [[nodiscard]] const glm::vec3 &getPosition() const;
    [[nodiscard]] const glm::vec3 &getDirection() const;
    [[nodiscard]] int getScreenWidth() const;
    [[nodiscard]] int getScreenHeight() const;
  };
}  // namespace nova::camera
#endif  // NOVA_CAMERA_H
