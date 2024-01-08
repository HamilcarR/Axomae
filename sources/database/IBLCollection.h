#ifndef IBLCOLLECTION_H
#define IBLCOLLECTION_H

#include "Axomae_macros.h"
#include "ImageDatabase.h"
#include "Observer.h"
#include "TextureDatabase.h"
#include "constants.h"
#include <vector>

/*Keeps track of baked envmaps and their ID in the texture database*/
struct EnvmapTextureGroup {
  /*TextureDatabase IDs*/
  int cubemap_id;
  int prefiltered_id;
  int irradiance_id;

  /*HdrImageDatabase ID*/
  int equirect_id;
};

class IBLCollection final : public ISubscriber {
 public:
  explicit IBLCollection(const TextureDatabase &texture_db, const HdrImageDatabase &image_db)
      : texture_database(texture_db), image_database(image_db) {}
  ~IBLCollection() = default;
  IBLCollection(IBLCollection &copy) = delete;
  IBLCollection(IBLCollection &&move) = delete;
  IBLCollection &operator=(IBLCollection &copy) = delete;
  IBLCollection &operator=(IBLCollection &&move) = delete;

 private:
  const TextureDatabase &texture_database;
  const HdrImageDatabase &image_database;
  std::vector<EnvmapTextureGroup> bakes_id;
};

#endif
