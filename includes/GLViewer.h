#ifndef GLVIEWER_H
#define GLVIEWER_H

#include <QMouseEvent> 

#include "constants.h"
#include "utils_3D.h" 
#include "Renderer.h" 
#include "Mesh.h"
#include <QOpenGLWidget>



/**
 * @file GLView.h
 * This file implements Viewer widget in the QT interface
 */

/**
 * @class GLViewer
 * This class implements methods for the drawing process
 */
class GLViewer : public QOpenGLWidget {

	Q_OBJECT
	
	public:

		/**
		 * @brief Construct a new GLViewer object
		 * 
		 * @param parent 
		 */
		GLViewer(QWidget* parent = nullptr); 
		
		/**
		 * @brief Destroy the GLViewer object
		 * 
		 */
		virtual ~GLViewer();	
		
		/**
		 * @brief Set the New Scene object
		 * 
		 * @param new_scene 
		 */
		virtual void setNewScene(std::pair<std::vector<Mesh*> , SceneTree> &new_scene);   
	protected:
		
		/**
		 * @brief Initialize GL context 
		 * 
		 */
		void initializeGL() override ; 
		
		/**
		 * @brief Draw method
		 * 
		 */
		void paintGL() override ; 
		
		/**
		 * @brief Resize the widget 
		 * 
		 * @param width New width in pixels
		 * @param height New height in pixels
		 */
		void resizeGL(int width , int height) override ; 
		
		/**
		 * @brief Displays info on the hardware and software used , and their versions
		 * 
		 */
		void printInfo() ; 
	private:
		
		/**
		 * @brief Mouse event triggered by moving the cursor  
		 * 
		 * @param event 
		 */
		void mouseMoveEvent(QMouseEvent *event) override; 
		
		/**
		 * @brief Mouse event triggered by click
		 * 
		 * @param event 
		 */
		void mousePressEvent(QMouseEvent *event) override; 
		
		/**
		 * @brief Mouse event triggered by releasing click 
		 * 
		 * @param event 
		 */
		void mouseReleaseEvent(QMouseEvent *event) override; 
		
		/**
		 * @brief Mouse event triggered by the mouse wheel 
		 * 
		 * @param event 
		 */
		void wheelEvent(QWheelEvent *event) override;
	private:
		Renderer* renderer; 		/*<Pointer on the renderer of the scene*/	
		bool glew_initialized ; 	/*<Check if context is initialized*/	






};




#endif
