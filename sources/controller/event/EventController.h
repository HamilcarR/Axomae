#ifndef EVENTCONTROLLER_H
#define EVENTCONTROLLER_H

class Event {

  enum TYPE : unsigned {
    /* Mouse events */
    EVENT_MOUSE_L = 1 << 0,
    EVENT_MOUSE_R = 1 << 1,
    EVENT_MOUSE_MIDDLE = 1 << 2,
    EVENT_MOUSE_L_DOUBLE = 1 << 3,
    EVENT_MOUSE_R_DOUBLE = 1 << 4,
    EVENT_MOUSE_SCROLL_UP = 1 << 5,
    EVENT_MOUSE_SCROLL_DOWN = 1 << 6,
    EVENT_MOUSE_MOVE = 1 << 7,
    /* Keyboard events */
  };

 public:
  int flag;
};

#endif  // EVENTCONTROLLER_H
