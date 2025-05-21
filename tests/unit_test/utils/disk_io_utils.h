#include "Image.h"
#include "ImageImporter.h"
#include "Loader.h"
#include "Metadata.h"
#include "Test.h"

template<class T>
static void write_texture_on_disk(const char *path,
                                  const std::vector<T> &texture,
                                  unsigned width,
                                  unsigned height,
                                  unsigned channels,
                                  const char *filetype,
                                  bool is_color_corrected,
                                  bool is_hdr) {
#ifdef DISK_WRITE
  image::Metadata metadata;
  metadata.channels = channels;
  metadata.width = width;
  metadata.height = height;
  metadata.format = filetype;
  metadata.color_corrected = is_color_corrected;
  metadata.is_hdr = is_hdr;
  image::ImageHolder<float> image(texture, metadata);
  IO::Loader loader(nullptr);
  loader.writeHdr("/tmp/envmaptest", image);
#endif
}
