#include "HdrListView.h"
#include "GUIWindow.h"

EnvmapListDisplay::EnvmapListDisplay(QWidget *parent) : QListView(parent), database(ResourceDatabaseManager::getInstance().getHdrDatabase()) {
  QListView::setModel(new HdrImageModel(database));
  QListView::setItemDelegate(new ThumbnailDelegate(this));
  QListView::setSelectionRectVisible(false);
  setVerticalScrollBar(new QScrollBar(this));
  connect(this, SIGNAL(clicked(const QModelIndex &)), SLOT(itemClicked(const QModelIndex &)));
}

void EnvmapListDisplay::itemClicked(const QModelIndex &index) { database.isSelected(index.row()); }