#include <internal/common/math/gpu/math_random_gpu.h>
#include <internal/common/math/math_random.h>
#include <nova/shape/mesh_transform_storage.h>
#include <unit_test/Test.h>

namespace transform = nova::shape::transform;

static glm::vec4 gen_rand_vec4(math::random::CPUPseudoRandomGenerator &gen) {
  return {gen.nrandf(-1000.f, 1000.f), gen.nrandf(-1000.f, 1000.f), gen.nrandf(-1000.f, 1000.f), gen.nrandf(-1000.f, 1000.f)};
}

static glm::mat4 gen_rand_mat4(math::random::CPUPseudoRandomGenerator &gen) {
  glm::mat4 result;
  for (int i = 0; i < 4; i++)
    result[i] = gen_rand_vec4(gen);
  return result;
}

TEST(transform4x4_t, hash_test) {
  transform4x4_t trfm{};
  math::random::CPUPseudoRandomGenerator gen;
  glm::mat4 m;
  for (int j = 0; j < 100; j++) {
    trfm.m = gen_rand_mat4(gen);
    std::size_t h0 = transform::hash(trfm);
    std::size_t h1 = transform::hash(trfm);
    ASSERT_EQ(h0, h1);
  }
}

TEST(Storage, getTransformOffset) {
  math::random::CPUPseudoRandomGenerator gen;
  transform::TransformStorage storage = transform::TransformStorage();
  glm::mat4 m = gen_rand_mat4(gen);
  glm::mat4 copy = m;
  storage.allocate(5);
  storage.add(m, 0);
  storage.add(m, 1);
  ASSERT_EQ(storage.getTransformOffset(m).transform_offset, storage.getTransformOffset(0));
  ASSERT_EQ(storage.getTransformOffset(m).transform_offset, storage.getTransformOffset(1));
  m = gen_rand_mat4(gen);
  storage.add(m, 2);
  ASSERT_EQ(storage.getTransformOffset(m).transform_offset, storage.getTransformOffset(2));
  m = gen_rand_mat4(gen);
  storage.add(m, 3);
  storage.add(copy, 4);
  ASSERT_EQ(storage.getTransformOffset(copy).transform_offset, storage.getTransformOffset(4));
  ASSERT_EQ(storage.getTransformOffset(m).transform_offset, storage.getTransformOffset(3));
}

TEST(matrix_element_storage_t, reconstruct_transform4x4_t) {
  math::random::CPUPseudoRandomGenerator gen;
  std::vector<glm::mat4> transforms;
  transforms.reserve(100);
  transform::TransformStorage storage = transform::TransformStorage();
  storage.allocate(100);
  for (int i = 0; i < 100; i++) {
    transforms.emplace_back(gen_rand_mat4(gen));
    storage.add(transforms.back(), i);
  }
  int offset = 0;
  transform4x4_t transform{};
  for (int i = 0; i < 100; i++) {

    bool valid = storage.reconstructTransform4x4(transform, offset);
    ASSERT_TRUE(valid);
    ASSERT_EQ(transform.m, transforms[i]);
    offset += transform4x4_t::padding();
  }
  ASSERT_FALSE(storage.reconstructTransform4x4(transform, offset));
}
