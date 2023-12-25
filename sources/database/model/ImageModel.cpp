#include "ImageModel.h"
#include "Image.h"
#include <QStandardItem>
HdrImageModel::HdrImageModel(ImageDatabase<float> &db, QObject *parent) : QAbstractListModel(parent), database(db), items_in_model(0) {
  database.attach(*this);
}

int HdrImageModel::rowCount(const QModelIndex &parent) const { return 3; }

int HdrImageModel::columnCount(const QModelIndex &parent) const { return 1; }

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
  }
}

QVariant HdrImageModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (role != Qt::DecorationRole)
    return {};
  if (orientation != Qt::Vertical)
    return {};
  return "name";
}

bool HdrImageModel::operator==(const ISubscriber<Message> &compare) const {
  if (this == &compare)
    return true;
  return false;
}

void HdrImageModel::notified(observer::Data<Message> message) {
  // beginResetModel();
  // setData(index(message.data.index, 0), QIcon(message.data.value));
  // endResetModel();
}
