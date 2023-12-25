#include "ImageModel.h"
#include <QStandardItem>
HdrImageModel::HdrImageModel(ImageDatabase<float> &db, QObject *parent) : QAbstractListModel(parent), database(db) {
  database.attach(*this);
  items_in_model = 0;
}

int HdrImageModel::rowCount(const QModelIndex &parent) const { return 3; }

int HdrImageModel::columnCount(const QModelIndex &parent) const { return 1; }

QVariant HdrImageModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid())
    return QVariant();
  if (index.row() >= database.size())
    return QVariant();
  QPixmap p = database.getThumbnail(index.row());
  if (role == Qt::DecorationRole)
    return p;
  if (role == Qt::SizeHintRole)
    return p.size();
  return QVariant();
}

QVariant HdrImageModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (role != Qt::DecorationRole)
    return QVariant();
  if (orientation != Qt::Vertical)
    return QVariant();
  return "name";
}

bool HdrImageModel::operator==(const ISubscriber<Message> &compare) const {
  if (this == &compare)
    return true;
  return false;
}

void HdrImageModel::notified(observer::Data<Message> message) {
  beginResetModel();
  setData(index(message.data.index, 0), QIcon(message.data.value));
  endResetModel();
}
