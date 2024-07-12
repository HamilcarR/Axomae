#ifndef IAXOBJECT_H
#define IAXOBJECT_H
#include "ILockable.h"
#include "OP_ProgressStatus.h"
class IAxObject : public ILockable, public controller::IProgressManager {};

#endif  // IAXOBJECT_H
