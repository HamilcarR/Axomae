#ifndef THUMBNAIL_H
#define THUMBNAIL_H
#include "constants.h"
#include <QImageWriter>
#include <QPixmap>

/**
 * @file ThumbnailList.h
 *
 */
class Thumbnail {
 public:
  Thumbnail() = default;
  Thumbnail(std::vector<float> &rgb, int width, int height, int channels = 3, bool normalize = false, int icon_width = 50, int icon_height = 50);

 public:
  QPixmap icon;
};

#endif