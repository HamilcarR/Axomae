#include "HdrListView.h"
#include "GUIWindow.h"

EnvmapListDisplay::EnvmapListDisplay(QWidget *parent) : QListView(parent), database(ResourceDatabaseManager::getInstance().getHdrDatabase()) {
  hdr_image_model = std::make_unique<HdrImageModel>(database);
  t_delegate = std::make_unique<ThumbnailDelegate>(this);
  QListView::setItemDelegate(t_delegate.get());
  QListView::setModel(hdr_image_model.get());
  QListView::setSelectionRectVisible(false);
  setVerticalScrollBar(new QScrollBar(this));
  connect(this, SIGNAL(clicked(const QModelIndex &)), SLOT(itemClicked(const QModelIndex &)));
}

void EnvmapListDisplay::itemClicked(const QModelIndex &index) {
  database.isSelected(index.row());
  gl_widget->update();
}

void EnvmapListDisplay::setWidget(GLViewer *widget) { gl_widget = widget; }