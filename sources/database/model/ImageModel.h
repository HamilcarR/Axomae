#ifndef IMAGEMODEL_H
#define IMAGEMODEL_H
#include "ImageDatabase.h"
#include "Observer.h"
#include "constants.h"
#include <QAbstractListModel>

class HdrImageModel : public QAbstractListModel, public ISubscriber<database::event::ImageUpdateMessage *> {

  Q_OBJECT

  using Message = database::event::ImageUpdateMessage *;

 public:
  explicit HdrImageModel(ImageDatabase<float> &db, QObject *parent = nullptr);
  [[nodiscard]] int rowCount(const QModelIndex &parent) const override;
  [[nodiscard]] QVariant data(const QModelIndex &index, int role) const override;
  [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
  void notified(observer::Data<Message> &data) override;

 protected:
  ImageDatabase<float> &database;
  int items_in_model;
};

#endif