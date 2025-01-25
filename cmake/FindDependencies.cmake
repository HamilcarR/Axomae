find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit)
find_package(Assimp QUIET)
find_package(Boost COMPONENTS 
  program_options 
  random stacktrace 
  container_hash
  math
)
