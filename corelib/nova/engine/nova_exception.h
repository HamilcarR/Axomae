#ifndef EXCEPTION_H
#define EXCEPTION_H

namespace nova::exception {

  enum ERROR : uint {
    NOERR = 0,
    INVALID_INTEGRATOR = 1 << 1,

    /* Sampler errors */
    INVALID_DIMENSION = 1 << 10,
    INVALID_SIZE = 1 << 11,

  };

  inline bool error_check(int err_flag) { return err_flag != NOERR; }
  /*inline std::vector<ERROR> get_error_list(int err_flag){}*/
}  // namespace nova::exception

#endif  // EXCEPTION_H
