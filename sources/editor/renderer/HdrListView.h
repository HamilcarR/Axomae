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
 public:
  explicit EnvmapListDisplay(QWidget *parent = nullptr) : QListView(parent), database(ResourceDatabaseManager::getInstance().getHdrDatabase()) {
    QListView::setModel(new HdrImageModel(database));
    QListView::setItemDelegate(new ThumbnailDelegate(this));
    QListView::setSelectionRectVisible(false);
    setVerticalScrollBar(new QScrollBar(this));
  }

  void mousePressEvent(QMouseEvent * /*event*/) override { ; }

 protected:
  HdrImageDatabase &database;
};

#endif