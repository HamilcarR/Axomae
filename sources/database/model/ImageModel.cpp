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
  const QPixmap &p = database.getThumbnail(index.row());
  image::Metadata metadata = database.getMetadata(index.row());
  switch (role) {
    case Qt::DecorationRole:
      return p;
    case Qt::DisplayRole:
      return QString(metadata.name.c_str());
    case Qt::SizeHintRole:
      return p.size();
    default:
      return {};
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

void HdrImageModel::notified(observer::Data<Message> &message) { emit layoutChanged(); }
