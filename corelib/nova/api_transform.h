#ifndef API_TRANSFORM_H
#define API_TRANSFORM_H
#include "api_common.h"

namespace nova {
  /**
   * @brief Abstract interface for 3D transformations
   * This interface provides methods for manipulating 3D transformations.
   */
  class Transform {
   public:
    virtual ~Transform() = default;

    /**
     * @brief Makes a clone Transform.
     *
     * @param other Transform to be cloned into this instance.
     */
    virtual void clone(const Transform &other) = 0;

    /**
     * @brief Set the transformation matrix from a 16-element float array
     *
     * @param transform Array of 16 floats representing a 4x4 transformation matrix
     *                  (column-major order)
     */
    virtual void setTransformMatrix(const float transform[16]) = 0;

    /**
     * @brief Apply rotation using quaternion parameters
     *
     * @param x X component of quaternion
     * @param y Y component of quaternion
     * @param z Z component of quaternion
     * @param w W component of quaternion
     */
    virtual void rotateQuat(float x, float y, float z, float w) = 0;

    /**
     * @brief Apply rotation using Euler angles around a specified axis
     *
     * @param angle Rotation angle in radians
     * @param x X component of rotation axis
     * @param y Y component of rotation axis
     * @param z Z component of rotation axis
     */
    virtual void rotateEuler(float angle, float x, float y, float z) = 0;

    /**
     * @brief Reset transformation to identity matrix
     */
    virtual void reset() = 0;

    /**
     * @brief Get the current transformation matrix as a 16-element float array
     *
     * @param transform Output array to store the 4x4 transformation matrix
     */
    virtual void getTransformMatrix(float transform[16]) const = 0;

    /**
     * @brief Apply translation to the current transformation
     *
     * @param x X translation component
     * @param y Y translation component
     * @param z Z translation component
     */
    virtual void translate(float x, float y, float z) = 0;

    /**
     * @brief Apply uniform scaling to the current transformation
     *
     * @param scale Uniform scale factor
     */
    virtual void scale(float scale) = 0;

    /**
     * @brief Apply non-uniform scaling to the current transformation
     *
     * @param x X scale component
     * @param y Y scale component
     * @param z Z scale component
     */
    virtual void scale(float x, float y, float z) = 0;

    /**
     * @brief Set the translation component of the transformation
     *
     * @param x X translation component
     * @param y Y translation component
     * @param z Z translation component
     */
    virtual void setTranslation(float x, float y, float z) = 0;

    /**
     * @brief Get the translation component of the transformation
     *
     * @param translation Output array to store translation vector [x, y, z]
     */
    virtual void getTranslation(float translation[3]) const = 0;

    /**
     * @brief Set the scale component of the transformation
     *
     * @param x X scale component
     * @param y Y scale component
     * @param z Z scale component
     */
    virtual void setScale(float x, float y, float z) = 0;

    /**
     * @brief Get the scale component of the transformation
     *
     * @param scale Output array to store scale vector [x, y, z]
     */
    virtual void getScale(float scale[3]) const = 0;

    /**
     * @brief Get the rotation component as a quaternion
     *
     * @param rotation Output array to store quaternion [x, y, z, w]
     */
    virtual void getRotation(float rotation[4]) const = 0;

    /**
     * @brief Set the rotation component using a quaternion
     *
     * @param rotation Rotation quaternion as array of 4 floats [x, y, z, w]
     */
    virtual void setRotation(const float rotation[4]) = 0;

    /**
     * @brief Multiply this transformation by another transformation
     *
     * @param other The transformation to multiply by
     */
    virtual void multiply(const Transform &other) = 0;

    /**
     * @brief Get the inverse of the current transformation
     *
     * @param inverse Output array to store the inverse transformation matrix [16 floats]
     */
    virtual void invert(float inverse[16]) const = 0;

    /**
     * @brief Get the transpose of the current transformation matrix
     *
     * @param result Output array to store the transposed transformation matrix [16 floats]
     */
    virtual void transpose(float result[16]) const = 0;

    /**
     * @brief Get the transpose of the inverse transformation matrix (useful for normal transformations)
     *
     * @param result Output array to store the transposed inverse matrix [9 floats for 3x3 matrix]
     */
    virtual void transposeInvert(float result[9]) const = 0;
  };

  TransformPtr create_transform();

}  // namespace nova

#endif
