//
// Created by hamilcar on 7/9/24.
//

#include "ContextMenuWidget.h"

#include <QMouseEvent>

ContextMenuWidget::ContextMenuWidget(QWidget *parent) : QMenu(parent) {}

ContextMenuWidget::ACTION ContextMenuWidget::spawnMenuOnPos(const QPoint &pos) {
  QMenu contextMenu(tr("Context menu"), this);

  QAction save("Save Render", this);
  QAction stop("Stop Sampling", this);

  contextMenu.addAction(&save);
  contextMenu.addAction(&stop);
  const QAction *result = contextMenu.exec(mapToGlobal(pos));
  if (result == &save)
    return SAVE;
  if (result == &stop)
    return STOP;
  else
    return NOACTION;
}
