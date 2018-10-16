#include "../includes/GUIWindow.h"
#include "../includes/Loader.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGraphicsItem>

namespace axomae {

	//TODO USE QLABEL FOR IMAGES
	enum IMAGETYPE : unsigned { GREYSCALE_LUMI = 1, HEIGHT = 2, NMAP = 3, DUDV = 4 , ALBEDO = 5 , GREYSCALE_AVG = 6 , PROJECTED_NMAP = 7}; 
	
	static double NORMAL_FACTOR = 1.;
	static float NORMAL_ATTENUATION = 1.56; 
	static double DUDV_FACTOR = 1.;
	static double dividor = 10; 
	template<typename T>
	struct image_type {
		T* image; 
		IMAGETYPE imagetype; 
	};

	/*structure to keep track of pointers to destroy*/

	class HeapManagement {
	public:
		 void addToHeap(image_type<SDL_Surface> a) {
			
		    std::vector<image_type<SDL_Surface>>::iterator it = std::find_if(SDLSurf_heap.begin(), SDLSurf_heap.end(), [a](image_type<SDL_Surface> b) { if (a.imagetype == b.imagetype) return true; else return false; });
			if (it != SDLSurf_heap.end()) {
					image_type<SDL_Surface> temp = *it;
					SDLSurf_heap.erase(it);
					SDLSurf_heap.push_back(a);
					SDL_FreeSurface(temp.image);
				}
			 else
				SDLSurf_heap.push_back(a);
		}
		 void addToTemp(image_type<SDL_Surface> a) {
			 temp_surfaces.push_back(a);
		 }
		 void addToHeap(image_type<QPaintDevice> a) {
			 std::vector<image_type<QPaintDevice>>::iterator it = std::find_if(paintDevice_heap.begin(), paintDevice_heap.end(), [a](image_type<QPaintDevice> b) { if (a.imagetype == b.imagetype) return true; else return false; });
			 if (it != paintDevice_heap.end()) {
				 image_type<QPaintDevice> temp = *it;
				 paintDevice_heap.erase(it);
				 paintDevice_heap.push_back(a);
				 delete temp.image; 
			 }
			 else
				 paintDevice_heap.push_back(a);
		 }
		 void addToHeap(image_type<QGraphicsItem> a) {
			 std::vector<image_type<QGraphicsItem>>::iterator it = std::find_if(graphicsItem_heap.begin(), graphicsItem_heap.end(), [a](image_type<QGraphicsItem> b) { if (a.imagetype == b.imagetype) return true; else return false; });
			 if (it != graphicsItem_heap.end()) {
				 image_type<QGraphicsItem> temp = *it;
				 graphicsItem_heap.erase(it);
				 graphicsItem_heap.push_back(a);
				 delete temp.image;
			 }
			 else
				 graphicsItem_heap.push_back(a);
		 }
		 void addToHeap(image_type<QObject> a) {
			 std::vector<image_type<QObject>>::iterator it = std::find_if(object_heap.begin(), object_heap.end(), [a](image_type<QObject> b) { if (a.imagetype == b.imagetype) return true; else return false; });
			 if (it != object_heap.end()) {
				 image_type<QObject> temp = *it;
				 object_heap.erase(it);
				 object_heap.push_back(a);
				 delete temp.image;
			 }
			 else
				 object_heap.push_back(a);
		 }

		 bool contain(IMAGETYPE A) {
			 auto lambda = [A](image_type<SDL_Surface> a) {
				 if (a.imagetype == A)
					 return true;
				 else
					 return false;

			 }; 
			return  std::any_of(SDLSurf_heap.begin(), SDLSurf_heap.end(), lambda); 
		}

		 

		~HeapManagement() {
				for (image_type<SDL_Surface> a : temp_surfaces) {
				SDL_FreeSurface(a.image);
				}
				for (image_type<SDL_Surface> a : SDLSurf_heap) {
				SDL_FreeSurface(a.image);
				}
				for (image_type<QPaintDevice> a : paintDevice_heap) {
					delete a.image;
					a.image = nullptr; 
				}
				for (image_type<QGraphicsItem> a : graphicsItem_heap) {
					delete a.image;
					a.image = nullptr;
				}
				for (image_type<QObject> a : object_heap) {
					delete a.image;
					a.image = nullptr;	
				}
		}
		image_type<SDL_Surface> getLastSurface() { return SDLSurf_heap.back();  }
	private:
		 std::vector<image_type<SDL_Surface>> SDLSurf_heap;
		 std::vector<image_type <QPaintDevice>> paintDevice_heap;
		 std::vector<image_type <QGraphicsItem>> graphicsItem_heap;
		 std::vector<image_type <QObject>> object_heap;
		 std::vector<image_type<SDL_Surface>> temp_surfaces;

		


	};

	/*structure fields pointing to the session datas*/
	namespace image_session_pointers {
		SDL_Surface* greyscale;
		SDL_Surface* albedo;
		SDL_Surface* height;
		SDL_Surface* normalmap;
		SDL_Surface* dudv;
		std::string filename; 
		void setAllNull() {
			filename = ""; 
			greyscale = nullptr;
			albedo = nullptr; 
			height = nullptr; 
			normalmap = nullptr;
			dudv = nullptr;
		}
	};
	
	
/****************************************************/
	HeapManagement *GUIWindow::_MemManagement = new HeapManagement;
	GUIWindow::GUIWindow( QWidget *parent) : QMainWindow(parent) {
		_UI.setupUi(this);
		connect_all_slots(); 
		_UI.progressBar->setValue(0);
		image_session_pointers::greyscale = nullptr; 
		image_session_pointers::albedo = nullptr; 
		image_session_pointers::height = nullptr; 
		image_session_pointers::normalmap = nullptr;
		image_session_pointers::dudv = nullptr; 

	}


	GUIWindow::~GUIWindow() {
		delete _MemManagement;
	}

/******************************************************************************************************************************************************************************************/

	static void display_image(SDL_Surface* surf , GUIWindow& window, QGraphicsView &view , IMAGETYPE type) {
		
		QImage *qimage = new QImage(static_cast<uchar*>(surf->pixels), surf->w, surf->h, QImage::Format_RGB888);
		QPixmap pix = QPixmap::fromImage(*qimage);
		QGraphicsPixmapItem * item = new QGraphicsPixmapItem(pix); //	std::shared_ptr<QGraphicsPixmapItem>  item(new QGraphicsPixmapItem(pix));
		auto scene = new QGraphicsScene(); //std::shared_ptr<QGraphicsScene> scene(new QGraphicsScene());
		scene->addItem(&*item);
		view.setScene(&*scene);
		view.fitInView(item);
		window._MemManagement->addToHeap({ qimage , type });
		window._MemManagement->addToHeap({ item , type });
		window._MemManagement->addToHeap({ scene , type });
		
	}





/******************************************************************************************************************************************************************************************/


	void GUIWindow::connect_all_slots() {
		QObject::connect(_UI.actionImport_image, SIGNAL(triggered()), this, SLOT(import_image()));
		QObject::connect(_UI.use_average, SIGNAL(clicked()), this, SLOT(greyscale_average()));
		QObject::connect(_UI.use_luminance, SIGNAL(clicked()), this, SLOT(greyscale_luminance()));
		QObject::connect(_UI.use_scharr, SIGNAL(clicked()), this, SLOT(use_scharr())); 
		QObject::connect(_UI.use_sobel, SIGNAL(clicked()), this, SLOT(use_sobel()));
		QObject::connect(_UI.use_prewitt, SIGNAL(clicked()), this, SLOT(use_prewitt()));
		QObject::connect(_UI.actionSave_image, SIGNAL(triggered()), this, SLOT(save_image()));

		QObject::connect(_UI.use_objectSpace, SIGNAL(clicked()), this, SLOT(use_object_space())); 
		QObject::connect(_UI.use_tangentSpace, SIGNAL(clicked()), this, SLOT(use_tangent_space()));
		
		QObject::connect(_UI.factor_slider_nmap, SIGNAL(valueChanged(int)), this, SLOT(change_nmap_factor(int)));
		QObject::connect(_UI.attenuation_slider_nmap, SIGNAL(valueChanged(int)), this, SLOT(change_nmap_attenuation(int)));

		QObject::connect(_UI.compute_dudv, SIGNAL(pressed()), this, SLOT(compute_dudv())); 
		QObject::connect(_UI.factor_slider_dudv, SIGNAL(valueChanged(int)), this, SLOT(change_dudv_nmap(int))); 
		QObject::connect(_UI.use_gpu, SIGNAL(clicked(bool)), this, SLOT(use_gpgpu(bool))); 

		QObject::connect(_UI.bake_texture,SIGNAL(clicked()) , this , SLOT(compute_projection())); 
		QObject::connect(_UI.actionImport_3D_model , SIGNAL(triggered()) , this , SLOT(import_3DOBJ())) ; 	
	
	
	
	
	}














/*SLOTS*/
/******************************************************************************************************************************************************************************************/
	bool GUIWindow::import_image() {
		HeapManagement* temp = _MemManagement; 
		_MemManagement = new HeapManagement; 
		QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), "./", tr("Images (*.png *.bmp *.jpg)")); 
		if (filename.isEmpty())
			return false;
		else {
			image_session_pointers::filename = filename.toStdString(); ;
			SDL_Surface* surf = ImageImporter::getInstance()->load_image(filename.toStdString().c_str()); 
			display_image(surf, *this, *_UI.diffuse_image , ALBEDO); 
			_MemManagement->addToHeap({ surf , ALBEDO });
			delete temp; 
			image_session_pointers::setAllNull(); 
			image_session_pointers::albedo = surf; 
			return true;

		}
	
	}

	/******************************************************************************************/
	bool GUIWindow::greyscale_average() {
		SDL_Surface* s = image_session_pointers::albedo; 
		SDL_Surface* copy = ImageManager::copy_surface(s); 

		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , GREYSCALE_AVG });
			ImageManager::set_greyscale_average(copy, 3);
			display_image(copy, *this, *_UI.greyscale_image , GREYSCALE_AVG);
			image_session_pointers::greyscale = copy; 
			return true; 
		}
		else
			return false;
	}

	/******************************************************************************************/

	bool GUIWindow::greyscale_luminance() {
		SDL_Surface* s = image_session_pointers::albedo; // use image_session_pointers TODO 
		SDL_Surface *copy = ImageManager::copy_surface(s);

		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , GREYSCALE_LUMI });
			ImageManager::set_greyscale_luminance(copy);
			display_image(copy, *this, *_UI.greyscale_image , GREYSCALE_LUMI);
			image_session_pointers::greyscale = copy;

			return true;
		}
		else
			return false;
	}

	/******************************************************************************************/

	void GUIWindow::use_gpgpu(bool checked) {
		if (checked)
			ImageManager::USE_GPU_COMPUTING();
		else
			ImageManager::USE_CPU_COMPUTING(); 

	}

	/******************************************************************************************/


	void GUIWindow::use_scharr() {
		SDL_Surface* s = image_session_pointers::greyscale; 
		SDL_Surface* copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , HEIGHT });
			ImageManager::compute_edge(copy, AXOMAE_USE_SCHARR, AXOMAE_REPEAT); 
			display_image(copy, *this, *_UI.height_image, HEIGHT); 
			image_session_pointers::height = copy; 
			

		}
		else {
			//TODO : error handling

		}

	}
	/******************************************************************************************/

	void GUIWindow::use_prewitt() {
		SDL_Surface* s = image_session_pointers::greyscale;
		SDL_Surface* copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , HEIGHT });
			ImageManager::compute_edge(copy, AXOMAE_USE_PREWITT, AXOMAE_REPEAT);
			display_image(copy, *this, *_UI.height_image, HEIGHT);
			image_session_pointers::height = copy;


		}
		else {
			//TODO : error handling

		}
	}
	/******************************************************************************************/

	void GUIWindow::use_sobel() {
		SDL_Surface* s = image_session_pointers::greyscale;
		SDL_Surface* copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , HEIGHT });
			ImageManager::compute_edge(copy, AXOMAE_USE_SOBEL, AXOMAE_REPEAT);
			display_image(copy, *this, *_UI.height_image, HEIGHT);
			image_session_pointers::height = copy;


		}
		else {
			//TODO : error handling

		}
	}


	/******************************************************************************************/


	void GUIWindow::use_tangent_space() {
		SDL_Surface* s = image_session_pointers::height;
		SDL_Surface* copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , NMAP });
			ImageManager::compute_normal_map(copy, NORMAL_FACTOR , NORMAL_ATTENUATION); 
			display_image(copy, *this, *_UI.normal_image, NMAP);
			image_session_pointers::normalmap = copy;


		}
		else {
			//TODO : error handling

		}

	}


	/******************************************************************************************/


	void GUIWindow::use_object_space() {

	}

	/******************************************************************************************/

	void GUIWindow::change_nmap_factor(int f) {
		NORMAL_FACTOR = f/dividor; 
		_UI.factor_nmap->setValue(NORMAL_FACTOR); 
		if (_UI.use_tangentSpace->isChecked()) {
			use_tangent_space();
		}
		else if (_UI.use_objectSpace->isChecked()) {
			use_object_space();
		}

	}

	void GUIWindow::change_nmap_attenuation(int f) {
		NORMAL_ATTENUATION = f / dividor;
		if (_UI.use_tangentSpace->isChecked()) {
			use_tangent_space(); 
		}
		else if (_UI.use_objectSpace->isChecked()) {
			use_object_space(); 
		}

	}
	/******************************************************************************************/

	void GUIWindow::compute_dudv() {
		SDL_Surface* s = image_session_pointers::normalmap;
		SDL_Surface* copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , DUDV });
			ImageManager::compute_dudv(copy, DUDV_FACTOR);
			display_image(copy, *this, *_UI.dudv_image, DUDV);
			image_session_pointers::dudv = copy;


		}
		else {
			//TODO : error handling

		}

	}
	/******************************************************************************************/
	void GUIWindow::change_dudv_nmap(int factor) {
		DUDV_FACTOR = factor / dividor;
		_UI.factor_dudv->setValue(DUDV_FACTOR);
		SDL_Surface* s = image_session_pointers::normalmap;
		SDL_Surface* copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			_MemManagement->addToHeap({ copy , DUDV });
			ImageManager::compute_dudv(copy, DUDV_FACTOR);
			display_image(copy, *this, *_UI.dudv_image, DUDV);
			image_session_pointers::dudv = copy;


		}
		else {
			//TODO : error handling

		}

	}
	/******************************************************************************************/
	void GUIWindow::compute_projection(){
		int width = _UI.uv_width->value() ; 
		int height = _UI.uv_height->value() ; 
		//SDL_Surface* s = ImageManager::	
		
	
	}

	/******************************************************************************************/
		


	/******************************************************************************************/
	
		
		


	/******************************************************************************************/

	bool GUIWindow::import_3DOBJ(){
		QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), "./", tr("3D models (*.obj *.fbx)"));
		if(!filename.isEmpty())
			{
				
				auto obj = Loader::load(filename.toStdString().c_str()) ;

				SDL_Surface* surf = ImageManager::project_uv_normals(obj[0] , _UI.uv_width->value() , _UI.uv_height->value()); 
				display_image(surf , *this , *_UI.uv_projection , PROJECTED_NMAP) ; 


			}

	}


	/******************************************************************************************/


	bool GUIWindow::open_project() {
		return false; 
	}
	/******************************************************************************************/

	bool GUIWindow::save_project() {
		return false;
	}
	/******************************************************************************************/

	bool GUIWindow::save_image() {
		ImageImporter *inst = ImageImporter::getInstance(); 
		QString filename = QFileDialog::getSaveFileName(this, tr("Save files"), "./", tr("All Files (*)"));
		if (image_session_pointers::height != nullptr)
			inst->save_image(image_session_pointers::height, (filename.toStdString()+"-height.bmp").c_str());
		if (image_session_pointers::normalmap != nullptr)
			inst->save_image(image_session_pointers::normalmap, (filename.toStdString() + "-nmap.bmp").c_str());
		if (image_session_pointers::dudv != nullptr)
			inst->save_image(image_session_pointers::dudv, (filename.toStdString() + "-dudv.bmp").c_str());

		return false;
	}
	
}
