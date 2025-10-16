#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "aggregate/aggregate_datastructures.h"
#include "primitive/NovaGeoPrimitive.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/Triangle.h"
#include "shape/mesh_transform_storage.h"
#include <unit_test/utils/geometry_shapes.h>
#include <unit_test/utils/mesh_geometry_utils.h>

class PrimitiveAggregateBuilder {
  Object3DBuilder mesh_builder;
  std::vector<nova::primitive::NovaPrimitiveInterface> primitives_collection;
  std::vector<nova::primitive::NovaGeoPrimitive> geometric_primitives;
  std::vector<nova::shape::NovaShapeInterface> shapes_collection;
  std::vector<nova::shape::Triangle> triangles_collection;
  std::vector<Object3D> mesh_list;
  /* May have a little bit of coupling , but it's way easier to represent transformations and their relations to meshes using Storage.*/
  nova::shape::transform::TransformStorage transform_storage;

 public:
  explicit PrimitiveAggregateBuilder(const glm::mat4 &transform);
  nova::aggregate::primitive_aggregate_data_s generateAggregate();

 private:
};

#endif
