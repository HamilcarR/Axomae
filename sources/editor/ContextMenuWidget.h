#ifndef CONTEXTMENUWIDGET_H
#define CONTEXTMENUWIDGET_H
#include "Axomae_macros.h"
#include <QMenu>

class ContextMenuWidget : public QMenu {
  Q_OBJECT
 public:
  enum ACTION : int { SAVE, STOP, NOACTION };

 public:
  explicit ContextMenuWidget(QWidget *parent = nullptr);

  ACTION spawnMenuOnPos(const QPoint &click_position);
};

#endif  // CONTEXTMENUWIDGET_H
