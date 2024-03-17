#ifndef EVENTCONTROLLER_H
#define EVENTCONTROLLER_H

namespace controller::event {

  struct MouseState {
    /* Screen position */
    int pos_x{0}, prev_pos_x{0};
    int pos_y{0}, prev_pos_y{0};
    /* Wheel scroll */
    int wheel_delta;
    bool busy{false};
  };

  class Event {
   public:
    enum TYPE : unsigned {
      NO_EVENT = 0,
      /* Mouse events */
      EVENT_MOUSE_L_PRESS = 1 << 0,
      EVENT_MOUSE_R_PRESS = 1 << 1,
      EVENT_MOUSE_L_RELEASE = 1 << 2,
      EVENT_MOUSE_R_RELEASE = 1 << 3,
      EVENT_MOUSE_MIDDLE_PRESS = 1 << 4,
      EVENT_MOUSE_WHEEL = 1 << 5,
      EVENT_MOUSE_L_DOUBLE = 1 << 6,
      EVENT_MOUSE_R_DOUBLE = 1 << 7,
      EVENT_MOUSE_MOVE = 1 << 8,
      /* Keyboard events */
    };

    unsigned long long flag;
    MouseState mouse_state{};
  };
}  // namespace controller::event
#endif  // EVENTCONTROLLER_H
