#ifndef EVENTINTERFACE_H
#define EVENTINTERFACE_H

namespace controller::event {
  class Event;
}

class EventInterface {
 public:
  virtual ~EventInterface() = default;
  virtual void processEvent(const controller::event::Event *event) = 0;
};
#endif  // EventInterface_H
