#include "ImageModel.h"
#include "Image.h"
#include <QStandardItem>
HdrImageModel::HdrImageModel(ImageDatabase<float> &db, QObject *parent) : QAbstractListModel(parent), database(db), items_in_model(0) {
  database.attach(*this);
}

int HdrImageModel::rowCount(const QModelIndex &parent) const {
  if (!parent.isValid()) {
    return database.size();
  } else
    return 0;
}

QVariant HdrImageModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid())
    return {};
  if (index.row() >= database.size())
    return {};
  QPixmap p = database.getThumbnail(index.row());
  image::Metadata metadata = database.getMetadata(index.row());
  switch (role) {
    case Qt::DecorationRole:
      return p;
      break;
    case Qt::DisplayRole:
      return QString(metadata.name.c_str());
      break;
    case Qt::SizeHintRole:
      return p.size();
      break;
    default:
      return {};
      break;
  }
}

QVariant HdrImageModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (role == Qt::DecorationRole) {
    if (orientation == Qt::Vertical) {
      return QString("Vertical");
    } else
      return QString("Horizontal");
  } else
    return {};
}

bool HdrImageModel::operator==(const ISubscriber<Message> &compare) const {
  if (this == &compare)
    return true;
  return false;
}

void HdrImageModel::notified(observer::Data<Message> message) { emit layoutChanged(); }
