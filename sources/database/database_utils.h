#ifndef DATABASE_UTILS_H
#define DATABASE_UTILS_H

#include "Image.h"
#include "constants.h"
#include <QModelIndex>
#include <QVariant>

namespace database::event {
  struct IconUpdateMessage {
    int index{};
    QPixmap *value{};
    image::Metadata metadata;
  };
}  // namespace database::event

#endif