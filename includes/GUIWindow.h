#ifndef GUIWINDOW_H
#define GUIWINDOW_H

#include <QtWidgets/qapplication.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qmainwindow.h>

#include "ImageImporter.h"
#include "ImageManager.h"
#include "Renderer.h"
#include "Window.h"
#include "../Form Files/test.h"
#include "constants.h"




namespace axomae {

	class HeapManagement; 
	class GUIWindow : public QMainWindow {
		Q_OBJECT
	public:

		GUIWindow( QWidget *parent = nullptr);
		~GUIWindow();
		Ui::MainWindow& getUi() { return _UI;  }

		static HeapManagement *_MemManagement;


	public slots:
	bool import_image(); 
	bool open_project(); 
	bool save_project(); 
	bool save_image();
	bool greyscale_average(); 
	bool greyscale_luminance();
	void use_scharr(); 
	void use_prewitt(); 
	void use_sobel(); 
	void use_gpgpu(bool checked); 
	void use_object_space(); 
	void use_tangent_space(); 
	void change_nmap_factor(int factor); 
	void change_nmap_attenuation(int atten); 
	void compute_dudv(); 
	void change_dudv_nmap(int factor);

	private:
		void connect_all_slots(); 


		Ui::MainWindow _UI;
		Renderer *_renderer; 
		Window *_window; 
		ImageImporter *_imporer; 
		

	};




}




#endif