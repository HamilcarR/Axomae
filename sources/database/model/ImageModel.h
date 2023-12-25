#ifndef IMAGEMODEL_H
#define IMAGEMODEL_H
#include "ImageDatabase.h"
#include "Observer.h"
#include "constants.h"
#include <QAbstractListModel>

class HdrImageModel : public QAbstractListModel, public ISubscriber<database::event::IconUpdateMessage> {

  Q_OBJECT

  using Message = database::event::IconUpdateMessage;

 public:
  explicit HdrImageModel(ImageDatabase<float> &db, QObject *parent = nullptr);
  int rowCount(const QModelIndex &parent) const;
  int columnCount(const QModelIndex &parent) const;
  QVariant data(const QModelIndex &index, int role) const override;
  QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
  bool operator==(const ISubscriber<Message> &compare) const final;
  void notified(observer::Data<Message> data) override;

 protected:
  ImageDatabase<float> &database;
  int items_in_model;
};

#endif