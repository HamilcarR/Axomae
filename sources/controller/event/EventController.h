#ifndef MOUSEEVENTCONTROLLER_H
#define MOUSEEVENTCONTROLLER_H

class EventController {

  enum MOUSEFLAG : unsigned {
    LEFT_CLICK = 1 << 0,
    RIGHT_CLICK = 1 << 1,
  };

 public:
  int flag;
};

#endif  // MOUSEEVENTCONTROLLER_H
