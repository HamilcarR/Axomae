#ifndef IMAGEMODEL_H
#define IMAGEMODEL_H
#include "ImageDatabase.h"
#include "Observer.h"
#include "constants.h"
#include <QAbstractListModel>

class HdrImageModel : public QAbstractListModel, public ISubscriber<database::event::Message> {

  Q_OBJECT

  using Message = database::event::Message;

 public:
  HdrImageModel(ImageDatabase<float> &db, QObject *parent = nullptr);
  int rowCount(const QModelIndex &parent = QModelIndex()) const;
  int columnCount(const QModelIndex &parent = QModelIndex()) const;
  QVariant data(const QModelIndex &index, int role) const override;
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  bool operator==(const ISubscriber<Message> &compare) const override final;
  void notified(observer::Data<Message> data) override final;

 protected:
  ImageDatabase<float> &database;
  int items_in_model;
};

#endif