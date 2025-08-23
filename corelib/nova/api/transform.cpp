#include "private_includes.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace nova {

  class NvTransform final : public Transform {
    glm::mat4 transform{};

   public:
    NvTransform() : transform(1.0f) {}

    void clone(const Transform &other) override {
      float temp_array[16] = {};
      other.getTransformMatrix(temp_array);
      setTransformMatrix(temp_array);
    }

    NvTransform(const Transform &transform) {
      float array[16];
      transform.getTransformMatrix(array);
      setTransformMatrix(array);
    }

    NvTransform &operator=(const Transform &transform) {
      float array[16];
      if (this != &transform) {
        transform.getTransformMatrix(array);
        setTransformMatrix(array);
      }
      return *this;
    }

    void setTransformMatrix(const float transform_array[16]) override {
      for (int i = 0; i < 16; i++)
        glm::value_ptr(transform)[i] = transform_array[i];
    }

    void rotateQuat(float x, float y, float z, float w) override {
      glm::quat quat(w, x, y, z);
      quat = glm::normalize(quat);
      glm::mat4 rotation_matrix = glm::mat4_cast(quat);
      transform = transform * rotation_matrix;
    }

    void rotateEuler(float angle, float x, float y, float z) override {
      glm::vec3 axis(x, y, z);
      if (glm::length(axis) > 0.0f) {
        axis = glm::normalize(axis);
        glm::mat4 rotation_matrix = glm::rotate(glm::mat4(1.0f), angle, axis);
        transform = transform * rotation_matrix;
      }
    }

    void reset() override { transform = glm::mat4(1.0f); }

    void getTransformMatrix(float transform_array[16]) const override {
      for (int i = 0; i < 16; i++)
        transform_array[i] = glm::value_ptr(transform)[i];
    }

    void translate(float x, float y, float z) override { transform = glm::translate(transform, glm::vec3(x, y, z)); }

    void scale(float scale) override { transform = glm::scale(transform, glm::vec3(scale)); }

    void scale(float x, float y, float z) override { transform = glm::scale(transform, glm::vec3(x, y, z)); }

    void setTranslation(float x, float y, float z) override {
      glm::vec3 scale_vec = glm::vec3(
          glm::length(glm::vec3(transform[0])), glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2])));

      transform = glm::mat4(1.0f);
      transform = glm::scale(transform, scale_vec);
      transform = glm::translate(transform, glm::vec3(x, y, z));
    }

    void getTranslation(float translation[3]) const override {
      glm::vec3 trans = glm::vec3(transform[3]);
      translation[0] = trans.x;
      translation[1] = trans.y;
      translation[2] = trans.z;
    }

    void setScale(float x, float y, float z) override {
      glm::vec3 translation = glm::vec3(transform[3]);

      transform = glm::mat4(1.0f);
      transform = glm::scale(transform, glm::vec3(x, y, z));
      transform = glm::translate(transform, translation);
    }

    void getScale(float scale[3]) const override {
      glm::vec3 scale_vec = glm::vec3(
          glm::length(glm::vec3(transform[0])), glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2])));
      scale[0] = scale_vec.x;
      scale[1] = scale_vec.y;
      scale[2] = scale_vec.z;
    }

    void getRotation(float rotation[4]) const override {
      glm::mat3 rot_mat = glm::mat3(transform);

      glm::quat quat = glm::quat_cast(rot_mat);

      rotation[0] = quat.x;
      rotation[1] = quat.y;
      rotation[2] = quat.z;
      rotation[3] = quat.w;
    }

    void setRotation(const float rotation[4]) override {
      glm::vec3 translation = glm::vec3(transform[3]);
      glm::vec3 scale_vec = glm::vec3(
          glm::length(glm::vec3(transform[0])), glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2])));

      glm::quat quat(rotation[3], rotation[0], rotation[1], rotation[2]);
      quat = glm::normalize(quat);

      transform = glm::mat4(1.0f);
      transform = glm::scale(transform, scale_vec);
      transform = glm::translate(transform, translation);
      transform = transform * glm::mat4_cast(quat);
    }

    void multiply(const Transform &other) override {
      float other_matrix[16];
      other.getTransformMatrix(other_matrix);

      glm::mat4 other_glm;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          other_glm[i][j] = other_matrix[i * 4 + j];
        }
      }
      transform = transform * other_glm;
    }

    void invert(float inverse[16]) const override {
      glm::mat4 inv_matrix = glm::inverse(transform);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          inverse[i * 4 + j] = inv_matrix[i][j];
        }
      }
    }

    void transpose(float result[16]) const override {
      glm::mat4 trans_matrix = glm::transpose(transform);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          result[i * 4 + j] = trans_matrix[i][j];
        }
      }
    }

    void transposeInvert(float result[9]) const override {
      glm::mat4 inv_matrix = glm::inverse(transform);
      glm::mat3 inv_3x3 = glm::mat3(inv_matrix);
      glm::mat3 trans_inv = glm::transpose(inv_3x3);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          result[i * 3 + j] = trans_inv[i][j];
        }
      }
    }
  };
  TransformPtr create_transform() { return std::make_unique<NvTransform>(); }
}  // namespace nova
