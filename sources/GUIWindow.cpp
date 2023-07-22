#include "../includes/GUIWindow.h"
#include "../includes/Loader.h"
#include "../includes/ImageImporter.h"
#include "../includes/ImageManager.h"
#include "../includes/Renderer.h"
#include "../includes/GLViewer.h" 
#include "../includes/SceneSelector.h" 
#include "../includes/MeshListView.h" 
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGraphicsItem>
#include <stack> 


namespace axomae { 
using namespace gui ; 

constexpr float POSTP_SLIDER_DIV = 90.f ; 
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
	
	void addToStack(image_type<SDL_Surface> a){
		image_type<SDL_Surface> copy = a ; 
		copy.image = GUIWindow::copy_surface(a.image);  
		SDLSurf_stack.push(copy); 		
	}
	image_type<SDL_Surface> topStack(){
		if(SDLSurf_stack.empty())
			return {nullptr , INVALID};  
		return SDLSurf_stack.top(); 
	}
	/* removes top element */
	void removeTopStack(){
		auto a = SDLSurf_stack.top(); 
		SDLSurf_stack.pop(); 
		SDLSurf_stack_temp.push(a); 
		//SDL_FreeSurface(a.image); 
	}
	void addTemptoStack(){
		if(!SDLSurf_stack_temp.empty()){
			auto a = SDLSurf_stack_temp.top(); 
			SDLSurf_stack_temp.pop() ; 
			SDLSurf_stack.push(a); 
		}
	}
	void clearStack(){
		while(!SDLSurf_stack.empty()){
				auto a = SDLSurf_stack.top(); 
				SDL_FreeSurface(a.image); 
				SDLSurf_stack.pop(); 
		}
	}
	void clearTempStack(){
		while(!SDLSurf_stack_temp.empty()){
				auto a = SDLSurf_stack_temp.top(); 
				SDL_FreeSurface(a.image); 
				SDLSurf_stack_temp.pop(); 
		}
	}
	void addToHeap(image_type<SDL_Surface> a) {
		addToStack(a); 
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
			SceneSelector::remove();
			clearStack(); 
			clearTempStack(); 
			for (image_type<SDL_Surface> a : temp_surfaces) 
				SDL_FreeSurface(a.image);
			for (image_type<SDL_Surface> a : SDLSurf_heap) 
				SDL_FreeSurface(a.image);
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

	image_type<SDL_Surface> getLastSurface() { 
		return SDLSurf_heap.back();
	}

private:
	std::stack<image_type<SDL_Surface>> SDLSurf_stack;
	std::stack<image_type<SDL_Surface>> SDLSurf_stack_temp; 
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
	SDL_Surface* uv_projection ; 
	std::string filename; 
	void setAllNull() {
		filename = "";
		uv_projection = nullptr ; 
		greyscale = nullptr;
		albedo = nullptr; 
		height = nullptr; 
		normalmap = nullptr;
		dudv = nullptr;
	}
}
		

/**************************************************************************************************************/

HeapManagement *GUIWindow::_MemManagement = new HeapManagement;

GUIWindow::GUIWindow( QWidget *parent) : QMainWindow(parent) {
	_UI.setupUi(this);
	connect_all_slots(); 
	_UI.progressBar->setValue(0);
	_renderer = _UI.renderer_view->getRenderer(); 
	image_session_pointers::greyscale = nullptr; 
	image_session_pointers::albedo = nullptr; 
	image_session_pointers::height = nullptr; 
	image_session_pointers::normalmap = nullptr;
	image_session_pointers::dudv = nullptr; 

}

GUIWindow::~GUIWindow() {
	delete _MemManagement;
}

/**************************************************************************************************************/
void GUIWindow::display_image(SDL_Surface* surf , IMAGETYPE type , bool save_in_heap) {
	QGraphicsView *view = get_corresponding_view(type);
	if(surf != nullptr && type != INVALID && save_in_heap){
		_MemManagement->addToHeap({surf , type}); 
		_MemManagement->clearTempStack(); 
	}
	if(view != nullptr){
		QImage *qimage = new QImage(static_cast<uchar*>(surf->pixels), surf->w, surf->h, QImage::Format_RGB888);
		QPixmap pix = QPixmap::fromImage(*qimage);
		QGraphicsPixmapItem * item = new QGraphicsPixmapItem(pix); 
		auto scene = new QGraphicsScene();
		scene->addItem(&*item);
		view->setScene(&*scene);
		view->fitInView(item);
		_MemManagement->addToHeap({ qimage , type });
		_MemManagement->addToHeap({ item , type });
		_MemManagement->addToHeap({ scene , type });
	}
}

/**************************************************************************************************************/
SDL_Surface* GUIWindow::copy_surface(SDL_Surface *src) {
	SDL_Surface* res; 
	res = SDL_CreateRGBSurface(src->flags, src->w, src->h, src->format->BitsPerPixel, src->format->Rmask, src->format->Gmask, src->format->Bmask, src->format->Amask); 
	if (res != nullptr) {
		SDL_BlitSurface(src, nullptr, res, nullptr); 
		return res;
	}
	else
		return nullptr; 
}
/**************************************************************************************************************/
QGraphicsView* GUIWindow::get_corresponding_view(IMAGETYPE type) {
	switch(type){
		case HEIGHT:
			return _UI.height_image; 
		break;
		case PROJECTED_NMAP:
			return _UI.uv_projection ; 
		break;	
		case ALBEDO:
			return _UI.diffuse_image ; 
		break;
		case GREYSCALE_LUMI :
			return _UI.greyscale_image ; 
		break ;
		case GREYSCALE_AVG:
			return _UI.greyscale_image; 
		break;
		case NMAP:
			return _UI.normal_image ; 
		break ; 
		case DUDV:
			return _UI.dudv_image ; 
		break ; 		
		default:
			return nullptr; 
		break; 
	}
}
/**************************************************************************************************************/
SDL_Surface* GUIWindow::get_corresponding_session_pointer(IMAGETYPE type){
	switch(type){
		case HEIGHT:
			return image_session_pointers::height; 
		break;

		case PROJECTED_NMAP:
			return image_session_pointers::uv_projection ; 
		break;
		
		case ALBEDO:
			return image_session_pointers::albedo ; 
		break;

		case GREYSCALE_LUMI :
			return image_session_pointers::greyscale; 
		break ;

		case GREYSCALE_AVG:
			return image_session_pointers::greyscale; 
		break;

		case NMAP:
			return image_session_pointers::normalmap ; 
		break ; 

		case DUDV:
			return image_session_pointers::dudv; 
		break ; 		
		
		default:
			return nullptr; 
		break; 
	}
}

/**************************************************************************************************************/
bool GUIWindow::set_corresponding_session_pointer(image_type<SDL_Surface> *image){
	switch(image->imagetype){
		case HEIGHT:
			image_session_pointers::height = image->image; 
		break;

		case PROJECTED_NMAP:
			image_session_pointers::uv_projection = image->image ; 
		break;
		
		case ALBEDO:
			image_session_pointers::albedo = image->image; 
		break;

		case GREYSCALE_LUMI :
			image_session_pointers::greyscale = image->image; 
		break ;

		case GREYSCALE_AVG:
			image_session_pointers::greyscale = image->image; 
		break;

		case NMAP:
			image_session_pointers::normalmap = image->image; 
		break ; 

		case DUDV:
			image_session_pointers::dudv = image->image; 
		break ; 		
		
		default:
			return false; 
		break; 
	}
	return true; 
}




/**************************************************************************************************************/
void GUIWindow::connect_all_slots() {
	QObject::connect(_UI.actionImport_image, SIGNAL(triggered()), this, SLOT(import_image()));
	QObject::connect(_UI.use_average, SIGNAL(clicked()), this, SLOT(greyscale_average()));
	QObject::connect(_UI.use_luminance, SIGNAL(clicked()), this, SLOT(greyscale_luminance()));
	QObject::connect(_UI.use_scharr, SIGNAL(clicked()), this, SLOT(use_scharr())); 
	QObject::connect(_UI.use_sobel, SIGNAL(clicked()), this, SLOT(use_sobel()));
	QObject::connect(_UI.use_prewitt, SIGNAL(clicked()), this, SLOT(use_prewitt()));
	QObject::connect(_UI.actionSave_image, SIGNAL(triggered()), this, SLOT(save_image()));
	QObject::connect(_UI.actionUndo , SIGNAL(triggered()) , this , SLOT(undo())); 
	QObject::connect(_UI.actionRedo , SIGNAL(triggered()) , this , SLOT(redo()));
	QObject::connect(_UI.undo_button , SIGNAL(clicked()) , this , SLOT(undo()));
	QObject::connect(_UI.redo_button , SIGNAL(clicked()) , this , SLOT(redo())); 
	QObject::connect(_UI.use_objectSpace, SIGNAL(clicked()), this, SLOT(use_object_space())); 
	QObject::connect(_UI.use_tangentSpace, SIGNAL(clicked()), this, SLOT(use_tangent_space()));
	
	QObject::connect(_UI.smooth_dial , SIGNAL(valueChanged(int)) , this , SLOT(update_smooth_factor(int))); 
	QObject::connect(_UI.sharpen_button , SIGNAL(clicked()) , this , SLOT(sharpen_edge())); 
	QObject::connect(_UI.smooth_button , SIGNAL(clicked()) , this , SLOT(smooth_edge())); 
	QObject::connect(_UI.factor_slider_nmap, SIGNAL(valueChanged(int)), this, SLOT(change_nmap_factor(int)));
	QObject::connect(_UI.attenuation_slider_nmap, SIGNAL(valueChanged(int)), this, SLOT(change_nmap_attenuation(int)));

	QObject::connect(_UI.compute_dudv, SIGNAL(pressed()), this, SLOT(compute_dudv())); 
	QObject::connect(_UI.factor_slider_dudv, SIGNAL(valueChanged(int)), this, SLOT(change_dudv_nmap(int))); 
	QObject::connect(_UI.use_gpu, SIGNAL(clicked(bool)), this, SLOT(use_gpgpu(bool))); 

	QObject::connect(_UI.bake_texture,SIGNAL(clicked()) , this , SLOT(compute_projection())); 
	QObject::connect(_UI.actionImport_3D_model , SIGNAL(triggered()) , this , SLOT(import_3DOBJ())) ; 	

	QObject::connect(_UI.next_mesh_button , SIGNAL(clicked()) , this , SLOT(next_mesh()));
	QObject::connect(_UI.previous_mesh_button , SIGNAL(clicked()) , this , SLOT(previous_mesh()));

	/*Renderer tab -> Post processing -> Camera*/
	QObject::connect(_UI.gamma_slider , SIGNAL(valueChanged(int)) , this , SLOT(set_renderer_gamma_value(int))); 	
	QObject::connect(_UI.exposure_slider , SIGNAL(valueChanged(int)) , this , SLOT(set_renderer_exposure_value(int))); 		
	QObject::connect(_UI.reset_camera_button , SIGNAL(pressed()) , this , SLOT(reset_renderer_camera())); 
	QObject::connect(_UI.set_standard_post_p , SIGNAL(clicked()) , this , SLOT(set_renderer_no_post_process())); 
	QObject::connect(_UI.set_edge_post_p , SIGNAL(pressed()) , this , SLOT(set_renderer_edge_post_process())); 
	QObject::connect(_UI.set_sharpen_post_p , SIGNAL(pressed()) , this , SLOT(set_renderer_sharpen_post_process())); 
	QObject::connect(_UI.set_blurr_post_p , SIGNAL(pressed()) , this , SLOT(set_renderer_blurr_post_process()));

	/*Renderer tab -> Rasterization -> Polygon display*/
	QObject::connect(_UI.rasterize_fill_button, SIGNAL(pressed()) , this , SLOT(set_rasterizer_fill())); 
	QObject::connect(_UI.rasterize_point_button , SIGNAL(pressed()) , this , SLOT(set_rasterizer_point()));
	QObject::connect(_UI.rasterize_wireframe_button , SIGNAL(pressed()) , this , SLOT(set_rasterizer_wireframe())); 
	QObject::connect(_UI.rasterize_display_bbox_checkbox , SIGNAL(toggled(bool)), this , SLOT(set_display_boundingbox(bool))); 



}


/*SLOTS*/
/**************************************************************************************************************/
bool GUIWindow::import_image() {
	HeapManagement* temp = _MemManagement; 
	_MemManagement = new HeapManagement; 
	QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), "./", tr("Images (*.png *.bmp *.jpg)")); 
	if (filename.isEmpty())
		return false;
	else {
		image_session_pointers::filename = filename.toStdString(); ;
		SDL_Surface* surf = ImageImporter::getInstance()->load_image(filename.toStdString().c_str()); 
		display_image(surf,  ALBEDO , true); 
		//_MemManagement->addToHeap({ surf , ALBEDO });
		delete temp; 
		image_session_pointers::setAllNull(); 
		image_session_pointers::albedo = surf; 
		return true;

	}

}

/**************************************************************************************************************/
bool GUIWindow::greyscale_average() {
	SDL_Surface* s = image_session_pointers::albedo; 
	SDL_Surface* copy = GUIWindow::copy_surface(s); 

	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , GREYSCALE_AVG });
		ImageManager::set_greyscale_average(copy, 3);
		display_image(copy,  GREYSCALE_AVG , true);
		image_session_pointers::greyscale = copy; 
		return true; 
	}
	else
		return false;
}

/**************************************************************************************************************/

bool GUIWindow::greyscale_luminance() {
	SDL_Surface* s = image_session_pointers::albedo; // use image_session_pointers TODO 
	SDL_Surface *copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , GREYSCALE_LUMI });
		ImageManager::set_greyscale_luminance(copy);
		display_image(copy,  GREYSCALE_LUMI , true);
		image_session_pointers::greyscale = copy;

		return true;
	}
	else
		return false;
}

/**************************************************************************************************************/

void GUIWindow::use_gpgpu(bool checked) {
	if (checked)
		ImageManager::USE_GPU_COMPUTING();
	else
		ImageManager::USE_CPU_COMPUTING(); 

}

/**************************************************************************************************************/


void GUIWindow::use_scharr() {
	SDL_Surface* s = image_session_pointers::greyscale; 
	SDL_Surface* copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , HEIGHT });
		ImageManager::compute_edge(copy, AXOMAE_USE_SCHARR, AXOMAE_REPEAT); 
		display_image(copy,  HEIGHT , true); 
		image_session_pointers::height = copy; 
		

	}
	else {
		//TODO : error handling

	}

}

/**************************************************************************************************************/
void GUIWindow::use_prewitt() {
	SDL_Surface* s = image_session_pointers::greyscale;
	SDL_Surface* copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , HEIGHT });
		ImageManager::compute_edge(copy, AXOMAE_USE_PREWITT, AXOMAE_REPEAT);
		display_image(copy,  HEIGHT , true);
		image_session_pointers::height = copy;


	}
	else {
		//TODO : error handling

	}
}

/**************************************************************************************************************/
void GUIWindow::use_sobel() {
	SDL_Surface* s = image_session_pointers::greyscale;
	SDL_Surface* copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , HEIGHT });
		ImageManager::compute_edge(copy, AXOMAE_USE_SOBEL, AXOMAE_REPEAT);
		display_image(copy,  HEIGHT , true);
		image_session_pointers::height = copy;


	}
	else {
		//TODO : error handling

	}
}



/**************************************************************************************************************/

void GUIWindow::use_tangent_space() {
	SDL_Surface* s = image_session_pointers::height;
	SDL_Surface* copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , NMAP });
		ImageManager::compute_normal_map(copy, NORMAL_FACTOR , NORMAL_ATTENUATION); 
		display_image(copy,  NMAP , true);
		image_session_pointers::normalmap = copy;


	}
	else {
		//TODO : error handling

	}

}


/**************************************************************************************************************/


void GUIWindow::use_object_space() {

}

/**************************************************************************************************************/

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

/**************************************************************************************************************/
void GUIWindow::compute_dudv() {
	SDL_Surface* s = image_session_pointers::normalmap;
	SDL_Surface* copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , DUDV });
		ImageManager::compute_dudv(copy, DUDV_FACTOR);
		display_image(copy,  DUDV , true);
		image_session_pointers::dudv = copy;


	}
	else {
		//TODO : error handling

	}

}

/**************************************************************************************************************/
void GUIWindow::change_dudv_nmap(int factor) {
	DUDV_FACTOR = factor / dividor;
	_UI.factor_dudv->setValue(DUDV_FACTOR);
	SDL_Surface* s = image_session_pointers::normalmap;
	SDL_Surface* copy = GUIWindow::copy_surface(s);
	if (copy != nullptr) {
		//_MemManagement->addToHeap({ copy , DUDV });
		ImageManager::compute_dudv(copy, DUDV_FACTOR);
		display_image(copy,  DUDV , true);
		image_session_pointers::dudv = copy;
	}
	else {
		//TODO : error handling

	}

}

/**************************************************************************************************************/
void GUIWindow::compute_projection(){ //TODO : complete uv projection method , implement baking
	int width = _UI.uv_width->value() ; 
	int height = _UI.uv_height->value() ; 
	width = 0 ; 
	height = 0 ; 
	
}

/**************************************************************************************************************/

//TODO: [AX-26] Optimize the normals projection on UVs in the UV tool 
void GUIWindow::project_uv_normals(){	
	SceneSelector* instance = SceneSelector::getInstance();
	Mesh* retrieved_mesh = instance->getCurrent() ; 
	if(retrieved_mesh){
		SDL_Surface* surf = ImageManager::project_uv_normals(retrieved_mesh->geometry , _UI.uv_width->value() , _UI.uv_height->value() , _UI.tangent_space->isChecked()); //TODO : change for managing the entire scene , maybe add scroll between different meshes 	
		display_image(surf , PROJECTED_NMAP , true);
	} 
}
/**************************************************************************************************************/
bool GUIWindow::import_3DOBJ(){
	QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), "./", tr("3D models (*.obj *.fbx *.glb)"));
	if(!filename.isEmpty()){
		Loader loader ;
		auto struct_holder =  loader.load(filename.toStdString().c_str());
		std::vector<Mesh*> scene = struct_holder.first;
		SceneSelector *instance = SceneSelector::getInstance(); 
		_UI.renderer_view->setNewScene(struct_holder); 
		instance->setScene(scene);
		_UI.meshes_list->setList(scene) ; 	
	//	std::thread(ImageManager::project_uv_normals, scene[0]->geometry , _UI.uv_width->value() , _UI.uv_height->value() , _UI.tangent_space->isChecked()).detach();  //TODO : optimize and re enable	
		return true ; 
	}
	return false ; 
}
/**************************************************************************************************************/
void GUIWindow::next_mesh(){
	SceneSelector::getInstance()->toNext();
	project_uv_normals(); 
}
/**************************************************************************************************************/
void GUIWindow::previous_mesh(){
	SceneSelector::getInstance()->toPrevious();
	project_uv_normals(); 
}

/**************************************************************************************************************/
bool GUIWindow::open_project() {
	return false; 
}

/**************************************************************************************************************/
bool GUIWindow::save_project() {
	return false;
}

/**************************************************************************************************************/
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

/**************************************************************************************************************/
void GUIWindow::smooth_edge(){
	SDL_Surface* surface = image_session_pointers::height;
	SDL_Surface* copy = GUIWindow::copy_surface(surface) ; 
	if (copy != nullptr) {
		unsigned int factor = _UI.smooth_dial->value(); 
		ImageManager::FILTER box_blur = _UI.box_blur_radio->isChecked() ? ImageManager::BOX_BLUR : ImageManager::FILTER_NULL ; 
		ImageManager::FILTER gaussian_blur_5_5 = _UI.gaussian_5_5_radio->isChecked() ? ImageManager::GAUSSIAN_SMOOTH_5_5 : ImageManager::FILTER_NULL ; 
		ImageManager::FILTER gaussian_blur_3_3 = _UI.gaussian_3_3_radio->isChecked() ? ImageManager::GAUSSIAN_SMOOTH_3_3 : ImageManager::FILTER_NULL ; 				
		ImageManager::smooth_image(copy, static_cast<ImageManager::FILTER> (box_blur | gaussian_blur_5_5 | gaussian_blur_3_3) , factor);
		display_image(copy, HEIGHT , true);
		image_session_pointers::height = copy ; 
	}
}

/**************************************************************************************************************/
void GUIWindow::sharpen_edge(){
	SDL_Surface* surface = image_session_pointers::greyscale;
	SDL_Surface* copy = GUIWindow::copy_surface(surface) ; 
	if (copy != nullptr) {
		float factor = _UI.sharpen_float_box->value(); 
		ImageManager::FILTER sharpen = _UI.sharpen_radio->isChecked() ? ImageManager::SHARPEN : ImageManager::FILTER_NULL ; 
		ImageManager::FILTER unsharp_masking = _UI.sharpen_masking_radio->isChecked() ? ImageManager::UNSHARP_MASKING : ImageManager::FILTER_NULL ; 
		ImageManager::sharpen_image(copy, static_cast<ImageManager::FILTER>(sharpen | unsharp_masking) , factor);
		display_image(copy, HEIGHT , true);
		image_session_pointers::height = copy ; 
	}

}
/**************************************************************************************************************/

void GUIWindow::undo(){
	image_type<SDL_Surface> previous = _MemManagement->topStack(); 
	if(previous.image != nullptr && previous.imagetype != INVALID){
		display_image(previous.image , previous.imagetype , false); 
		_MemManagement->removeTopStack(); 
		set_corresponding_session_pointer(&previous) ; 
	}

}

/**************************************************************************************************************/

void GUIWindow::redo(){
	//TODO: [AX-41] Fix crash when processing using undo / redo values on the Stack
	_MemManagement->addTemptoStack() ; 
	image_type<SDL_Surface> next = _MemManagement->topStack(); 
	if(next.image != nullptr && next.imagetype != INVALID){
		display_image(next.image , next.imagetype , false); 
		set_corresponding_session_pointer(&next); 
	}

}

/**************************************************************************************************************/

void GUIWindow::set_renderer_gamma_value(int value){
	if(_renderer != nullptr){
		float v = (float) value / POSTP_SLIDER_DIV ; 
		_renderer->setGammaValue(v); 
	}
}

/**************************************************************************************************************/

void GUIWindow::reset_renderer_camera(){
	if(_renderer != nullptr)
		_renderer->resetSceneCamera();
}


/**************************************************************************************************************/

void GUIWindow::set_renderer_exposure_value(int value){
	if(_renderer != nullptr){
		float v = (float) value / POSTP_SLIDER_DIV ; 
		_renderer->setExposureValue(v); 
	}
}
/**************************************************************************************************************/
void GUIWindow::set_renderer_no_post_process(){
	if(_renderer != nullptr){
		_renderer->setNoPostProcess();
	}
} 
/**************************************************************************************************************/
void GUIWindow::set_renderer_edge_post_process(){
	if(_renderer != nullptr){
		_renderer->setPostProcessEdge(); 
	}
}

/**************************************************************************************************************/
void GUIWindow::set_renderer_sharpen_post_process(){
	if(_renderer != nullptr)
		_renderer->setPostProcessSharpen(); 
}

/**************************************************************************************************************/
void GUIWindow::set_renderer_blurr_post_process(){
	if(_renderer != nullptr)
		_renderer->setPostProcessBlurr(); 
}

/**************************************************************************************************************/
void GUIWindow::set_rasterizer_point(){
	if(_renderer)
		_renderer->setRasterizerPoint(); 
}

/**************************************************************************************************************/
void GUIWindow::set_rasterizer_fill(){
	if(_renderer)
		_renderer->setRasterizerFill(); 
}

/**************************************************************************************************************/
void GUIWindow::set_rasterizer_wireframe(){
	if(_renderer)
		_renderer->setRasterizerWireframe(); 
} 

/**************************************************************************************************************/
void GUIWindow::set_display_boundingbox(bool display){
	if(_renderer)
		_renderer->displayBoundingBoxes(display); 
} 










/**************************************************************************************************************/
/*Protected utility methods*/

void GUIWindow::update_smooth_factor(int factor){
	_UI.smooth_factor->setValue(factor); 
}




/*
 * TODO: add custom QGraphicsView class , reimplement resizeEvent() to scale GraphicsView to window size */























































}

