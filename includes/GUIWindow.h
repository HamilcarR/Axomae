#ifndef GUIWINDOW_H
#define GUIWINDOW_H

#include <QtWidgets/qapplication.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qmainwindow.h>

#include "Window.h"
#include "../Form Files/ui_test.h"
#include "SceneSelector.h" 
#include "constants.h"
#include "utils_3D.h" 


namespace gui {
	enum IMAGETYPE : unsigned { GREYSCALE_LUMI = 1, HEIGHT = 2, NMAP = 3, DUDV = 4 , ALBEDO = 5 , GREYSCALE_AVG = 6 , PROJECTED_NMAP = 7 , INVALID = 8}; 
}

namespace axomae {
	class HeapManagement; 
	class Renderer ; 
	class ImageImporter ;
	template <typename T> struct image_type ;  		
	class GUIWindow : public QMainWindow {

		Q_OBJECT
	public:
	
		GUIWindow( QWidget *parent = nullptr);
		~GUIWindow();
		Ui::MainWindow& getUi() { return _UI;  }
		static HeapManagement *_MemManagement;
		static SDL_Surface* copy_surface(SDL_Surface* surface); 	
	/* SLOTS */
	public slots:
		bool import_image(); 
		bool import_3DOBJ(); 
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
		void compute_projection() ; 
		void next_mesh(); 
		void previous_mesh(); 
		void project_uv_normals();
		void smooth_edge(); 
		void sharpen_edge(); 
		void undo();
		void redo();


	protected slots:
		void update_smooth_factor(int factor);
	private:
		void connect_all_slots(); 
		QGraphicsView* get_corresponding_view(gui::IMAGETYPE image);
		SDL_Surface* get_corresponding_session_pointer(gui::IMAGETYPE image);
		bool set_corresponding_session_pointer(image_type<SDL_Surface> *image_type_pointer) ; 
		void display_image(SDL_Surface *surf , gui::IMAGETYPE image , bool save_in_heap); 
		Ui::MainWindow _UI;
		Renderer *_renderer; 
		Window *_window; 
		ImageImporter *_importer; 
		
	};




}




#endif
