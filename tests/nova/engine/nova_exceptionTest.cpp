#include "engine/nova_exception.h"
#include "Test.h"

static bool is_present(const std::vector<nova::exception::ERROR> &error_vector, nova::exception::ERROR error) {
  for (const auto &elem : error_vector)
    if (error == elem)
      return true;
  return false;
}

TEST(nova_exception_test, get_error_list) {
  nova::exception::NovaException exception;
  exception.addErrorType(nova::exception::SAMPLER_INIT_ERROR | nova::exception::INVALID_SAMPLER_DIM);
  auto error_list = exception.getErrorList();
  ASSERT_TRUE(is_present(error_list, nova::exception::SAMPLER_INIT_ERROR));
  ASSERT_TRUE(is_present(error_list, nova::exception::INVALID_SAMPLER_DIM));
}

TEST(nova_exception_test, add_error_type) {
  nova::exception::NovaException exception;
  exception.addErrorType(nova::exception::SAMPLER_INIT_ERROR);
  exception.addErrorType(nova::exception::INVALID_SAMPLER_DIM);
  auto flag = exception.getErrorFlag();
  ASSERT_TRUE(flag & nova::exception::SAMPLER_INIT_ERROR);
  ASSERT_TRUE(flag & nova::exception::INVALID_SAMPLER_DIM);
}

TEST(nova_exception_test, copy_constructor) {
  nova::exception::NovaException origin;
  origin.addErrorType(nova::exception::INVALID_INTEGRATOR | nova::exception::SAMPLER_INIT_ERROR | nova::exception::INVALID_RENDER_MODE);
  nova::exception::NovaException target;
  target = origin;
  auto err_list = target.getErrorList();
  ASSERT_TRUE(is_present(err_list, nova::exception::INVALID_INTEGRATOR));
  ASSERT_TRUE(is_present(err_list, nova::exception::SAMPLER_INIT_ERROR));
  ASSERT_TRUE(is_present(err_list, nova::exception::INVALID_RENDER_MODE));
  ASSERT_FALSE(is_present(err_list, nova::exception::INVALID_SAMPLER_DIM));
}

TEST(nova_exception_test, move_constructor) {
  nova::exception::NovaException origin;
  origin.addErrorType(nova::exception::INVALID_INTEGRATOR | nova::exception::SAMPLER_INIT_ERROR | nova::exception::INVALID_RENDER_MODE);
  nova::exception::NovaException target;
  target = std::move(origin);
  auto err_list = target.getErrorList();
  ASSERT_TRUE(is_present(err_list, nova::exception::INVALID_INTEGRATOR));
  ASSERT_TRUE(is_present(err_list, nova::exception::SAMPLER_INIT_ERROR));
  ASSERT_TRUE(is_present(err_list, nova::exception::INVALID_RENDER_MODE));
  ASSERT_FALSE(is_present(err_list, nova::exception::INVALID_SAMPLER_DIM));
}

TEST(nova_exception_test, merge) {
  nova::exception::NovaException origin;
  origin.addErrorType(nova::exception::SAMPLER_INIT_ERROR | nova::exception::INVALID_SAMPLER_DIM);
  nova::exception::NovaException target;
  target.addErrorType(nova::exception::SAMPLER_INVALID_ARG | nova::exception::SAMPLER_INVALID_ALLOC);
  int a = target.merge(origin.getErrorFlag());
  auto err_list = target.getErrorList();
  ASSERT_TRUE(is_present(err_list, nova::exception::SAMPLER_INIT_ERROR));
  ASSERT_TRUE(is_present(err_list, nova::exception::INVALID_SAMPLER_DIM));
  ASSERT_TRUE(is_present(err_list, nova::exception::SAMPLER_INVALID_ARG));
  ASSERT_TRUE(is_present(err_list, nova::exception::SAMPLER_INVALID_ALLOC));
  ASSERT_FALSE(is_present(err_list, nova::exception::INVALID_RENDER_MODE));
}