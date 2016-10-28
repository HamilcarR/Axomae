#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "Model.h"
#include "View.h"




 namespace maptomix {

class Controller{
	public:
		Controller();
		Controller(View* view,Model* data);
		~Controller();




	private:
		View *view;
		Model *model;


};

















}














#endif
