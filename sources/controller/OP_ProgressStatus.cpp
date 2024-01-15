//
// Created by hamilcar on 1/15/24.
//

#include "OP_ProgressStatus.h"

namespace controller {
  namespace progress_bar {

    OP_ProgressStatus::OP_ProgressStatus(ProgressStatusWidget *progress_bar) { widget = progress_bar; }

    bool OP_ProgressStatus::op() const {}
  }  // namespace progress_bar
}  // namespace controller