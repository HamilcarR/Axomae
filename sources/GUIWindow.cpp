#include "../includes/GUIWindow.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGraphicsItem>

namespace axomae {

	/*structure to keep track of pointers to destroy*/

	class HeapManagement {
	public:
		 void addToHeap(SDL_Surface* a) {
			SDLSurf_heap.push_back(a);
		}
		 void addToHeap(QPaintDevice* a) {
			paintDevice_heap.push_back(a);
		}
		 void addToHeap(QGraphicsItem* a) {
			graphicsItem_heap.push_back(a);
		}
		 void addToHeap(QObject* a) {
			object_heap.push_back(a);
		}
		~HeapManagement() {
			for (SDL_Surface* a : SDLSurf_heap)
				SDL_FreeSurface(a);

			for (QPaintDevice* a : paintDevice_heap)
				delete a;

			for (QGraphicsItem *a : graphicsItem_heap)
				delete a;

			for (QObject *a : object_heap)
				delete a;
		}
		SDL_Surface* getLastSurface() { return SDLSurf_heap.back();  }
	private:
		 std::vector<SDL_Surface*> SDLSurf_heap;
		 std::vector<QPaintDevice*> paintDevice_heap;
		 std::vector<QGraphicsItem*> graphicsItem_heap;
		 std::vector<QObject*> object_heap;


	};
	
/****************************************************/
	HeapManagement *GUIWindow::_MemManagement = new HeapManagement;
	GUIWindow::GUIWindow( QWidget *parent) : QMainWindow(parent) {
		_UI.setupUi(this);
		connect_all_slots(); 
		_UI.progressBar->setValue(0); 

	}


	GUIWindow::~GUIWindow() {
		delete _MemManagement;
	}

/******************************************************************************************************************************************************************************************/

	static void display_image(SDL_Surface* surf , GUIWindow& window, QGraphicsView &view) {
		QImage *qimage = new QImage(static_cast<uchar*>(surf->pixels), surf->w, surf->h, QImage::Format_RGB888);
		QPixmap pix = QPixmap::fromImage(*qimage);
		QGraphicsPixmapItem * item = new QGraphicsPixmapItem(pix); //	std::shared_ptr<QGraphicsPixmapItem>  item(new QGraphicsPixmapItem(pix));
		auto scene = new QGraphicsScene(); //std::shared_ptr<QGraphicsScene> scene(new QGraphicsScene());
		scene->addItem(&*item);
		view.setScene(&*scene);

		window._MemManagement->addToHeap(qimage);
		window._MemManagement->addToHeap(item);
		window._MemManagement->addToHeap(scene);
		
	}





/******************************************************************************************************************************************************************************************/


	void GUIWindow::connect_all_slots() {
		QObject::connect(_UI.actionImport_image, SIGNAL(triggered()), this, SLOT(import_image()));
		QObject::connect(_UI.use_average, SIGNAL(clicked()), this, SLOT(greyscale_average()));
		QObject::connect(_UI.use_luminance, SIGNAL(clicked()), this, SLOT(greyscale_luminance()));

		QObject::connect(_UI.use_gpu, SIGNAL(clicked(bool)), this, SLOT(use_gpgpu(bool))); 

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
			SDL_Surface* surf = ImageImporter::getInstance()->load_image(filename.toStdString().c_str()); 
			display_image(surf, *this, *_UI.diffuse_image); 


			_MemManagement->addToHeap(surf); 
			
			delete temp; 
			
			return true;

		}
	
	}


	bool GUIWindow::greyscale_average() {
		SDL_Surface* s = _MemManagement->getLastSurface(); 
		SDL_Surface *copy = ImageManager::copy_surface(s);
		if (copy != nullptr) {
			ImageManager::set_greyscale_average(copy, 3);
			display_image(copy, *this, *_UI.greyscale_image);
			return true; 
		}
		else
			return false;
	}


	bool GUIWindow::greyscale_luminance() {
		SDL_Surface* s = _MemManagement->getLastSurface();
		SDL_Surface *copy = ImageManager::copy_surface(s); 

		if (copy != nullptr) {
			ImageManager::set_greyscale_luminance(copy);
			display_image(copy, *this, *_UI.greyscale_image);
			return true;
		}
		else
			return false;
	}


	void GUIWindow::use_gpgpu(bool checked) {
		if (checked)
			ImageManager::USE_GPU_COMPUTING();
		else
			ImageManager::USE_CPU_COMPUTING(); 

	}


	bool GUIWindow::open_project() {
		return false; 
	}
	bool GUIWindow::save_project() {
		return false;
	}
	bool GUIWindow::save_image() {
		return false;
	}



}