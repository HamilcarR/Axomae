#include "private_includes.h"
namespace nova {

  static void copy_camera(const Camera &cam, camera::CameraResourcesHolder &camera) {
    float vec3[3]{}, mat16[16]{};

    cam.getUpVector(vec3);
    camera.up_vector = f3_to_vec3(vec3);

    cam.getProjectionMatrix(mat16);
    camera.P = f16_to_mat4(mat16);

    cam.getViewMatrix(mat16);
    camera.V = f16_to_mat4(mat16);

    cam.getProjectionViewMatrix(mat16);
    camera.PV = f16_to_mat4(mat16);

    cam.getInverseProjectionViewMatrix(mat16);
    camera.inv_PV = f16_to_mat4(mat16);

    cam.getInverseProjectionMatrix(mat16);
    camera.inv_P = f16_to_mat4(mat16);

    cam.getInverseViewMatrix(mat16);
    camera.inv_V = f16_to_mat4(mat16);

    cam.getUpVector(vec3);
    camera.up_vector = f3_to_vec3(vec3);

    cam.getPosition(vec3);
    camera.position = f3_to_vec3(vec3);

    cam.getDirection(vec3);
    camera.direction = f3_to_vec3(vec3);

    camera.far = cam.getClipPlaneFar();
    camera.near = cam.getClipPlaneNear();
    camera.fov = cam.getFov();
    camera.screen_width = cam.getResolutionWidth();
    camera.screen_height = cam.getResolutionHeight();
  }

  class NvCamera final : public Camera {
    camera::CameraResourcesHolder camera{};

   public:
    NvCamera() = default;

    NvCamera &operator=(const Camera &cam) {
      if (this != &cam)
        copy_camera(cam, camera);
      return *this;
    }

    NvCamera(const Camera &cam) { copy_camera(cam, camera); }

    ERROR_STATE getProjectionViewMatrix(float proj[16]) const override {
      glm::mat4 pv;
      pv = camera.inv_PV;
      for (int i = 0; i < 16; i++)
        proj[i] = glm::value_ptr(pv)[i];
      return SUCCESS;
    }

    ERROR_STATE getInverseViewMatrix(float proj[16]) const override {
      glm::mat4 iv;
      iv = camera.inv_V;
      for (int i = 0; i < 16; i++)
        proj[i] = glm::value_ptr(iv)[i];
      return SUCCESS;
    }

    ERROR_STATE getInverseProjectionViewMatrix(float proj[16]) const override {
      glm::mat4 ipv;
      ipv = camera.inv_PV;
      for (int i = 0; i < 16; i++)
        proj[i] = glm::value_ptr(ipv)[i];
      return SUCCESS;
    }

    ERROR_STATE getInverseProjectionMatrix(float proj[16]) const override {
      glm::mat4 ip;
      ip = camera.inv_P;
      for (int i = 0; i < 16; i++)
        proj[i] = glm::value_ptr(ip)[i];
      return SUCCESS;
    }

    ERROR_STATE setUpVector(const float up[3]) override {
      camera.up_vector = glm::vec3(up[0], up[1], up[2]);
      return SUCCESS;
    }

    ERROR_STATE setMatrices(const float projection_matrix[16], const float view_matrix[16]) override {
      for (int i = 0; i < 16; i++)
        glm::value_ptr(camera.P)[i] = projection_matrix[i];
      for (int i = 0; i < 16; i++)
        glm::value_ptr(camera.V)[i] = view_matrix[i];

      camera.inv_P = glm::inverse(camera.P);
      camera.inv_V = glm::inverse(camera.V);
      camera.PV = camera.P * camera.V;
      camera.inv_PV = glm::inverse(camera.PV);
      return SUCCESS;
    }

    ERROR_STATE setPosition(const float pos[3]) override {
      camera.position = glm::vec3(pos[0], pos[1], pos[2]);
      return SUCCESS;
    }

    ERROR_STATE setDirection(const float dir[3]) override {
      camera.direction = glm::vec3(dir[0], dir[1], dir[2]);
      return SUCCESS;
    }

    ERROR_STATE setClipPlaneFar(float far_plane) override {
      camera.far = far_plane;
      return SUCCESS;
    }

    ERROR_STATE setClipPlaneNear(float near_plane) override {
      camera.near = near_plane;
      return SUCCESS;
    }

    ERROR_STATE setResolutionWidth(unsigned width) override {
      camera.screen_width = width;
      return SUCCESS;
    }

    ERROR_STATE setResolutionHeight(unsigned height) override {
      camera.screen_height = height;
      return SUCCESS;
    }

    ERROR_STATE setFov(float fov) override {
      camera.fov = fov;
      return SUCCESS;
    }

    ERROR_STATE getUpVector(float up[3]) const override {
      up[0] = camera.up_vector.x;
      up[1] = camera.up_vector.y;
      up[2] = camera.up_vector.z;
      return SUCCESS;
    }

    ERROR_STATE getProjectionMatrix(float proj[16]) const override {
      for (int i = 0; i < 16; i++)
        proj[i] = glm::value_ptr(camera.P)[i];
      return SUCCESS;
    }

    ERROR_STATE getViewMatrix(float view[16]) const override {
      for (int i = 0; i < 16; i++)
        view[i] = glm::value_ptr(camera.V)[i];
      return SUCCESS;
    }

    ERROR_STATE getPosition(float pos[3]) const override {
      pos[0] = camera.position.x;
      pos[1] = camera.position.y;
      pos[2] = camera.position.z;
      return SUCCESS;
    }

    ERROR_STATE getDirection(float dir[3]) const override {
      dir[0] = camera.direction.x;
      dir[1] = camera.direction.y;
      dir[2] = camera.direction.z;
      return SUCCESS;
    }

    float getClipPlaneFar() const override { return camera.far; }

    float getClipPlaneNear() const override { return camera.near; }

    unsigned getResolutionWidth() const override { return camera.screen_width; }

    unsigned getResolutionHeight() const override { return camera.screen_height; }

    float getFov() const override { return camera.fov; }
  };

  CameraPtr create_camera() { return std::make_unique<NvCamera>(); }

}  // namespace nova
