#ifndef DATABASE_UTILS_H
#define DATABASE_UTILS_H

#include "constants.h"
#include <QModelIndex>
#include <QVariant>
namespace database {
  namespace event {
    struct Message {
      int index;
      QPixmap value;
    };
  };  // namespace event

};  // namespace database

#endif