#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H
namespace nova {
   template<class T>
  struct RenderBuffers {
    T *accumulator_buffer;
    T *partial_buffer;
    size_t byte_size_buffers{};
    int channels{};
    T *depth_buffer;
  };
  using HdrBufferStruct = RenderBuffers<float>;
}
#endif //DATASTRUCTURES_H
