#ifndef HDRLISTVIEW_H
#define HDRLISTVIEW_H
#include "ImageDatabase.h"
#include "ImageModel.h"
#include "ResourceDatabaseManager.h"
#include <QHeaderView>
#include <QListView>
#include <QPainter>
#include <QScrollBar>
#include <QStyledItemDelegate>

class ThumbnailDelegate : public QStyledItemDelegate {
 public:
  explicit ThumbnailDelegate(QObject *parent = nullptr) : QStyledItemDelegate(parent) {}
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
    QStyledItemDelegate::paint(painter, option, index);
  }

  QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override { return QSize(100, 50); }
};

class EnvmapListDisplay : public QListView {
  Q_OBJECT
 public:
  explicit EnvmapListDisplay(QWidget *parent = nullptr);
 protected slots:
  void itemClicked(const QModelIndex &index);

 protected:
  HdrImageDatabase &database;
};

#endif