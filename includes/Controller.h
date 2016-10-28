#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "Data.h"
#include "View.h"
#include "EventHandler.h"




 namespace maptomix {

class Controller{
	public:
		Controller(View* view,Data* data);
		~Controller();




	private:
		View *view;
		Data *data;

		EventHandler* m_event_handler;

};

















}














#endif
