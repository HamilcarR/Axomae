/********************************************************************************
** Form generated from reading UI file 'main_window.ui'
**
** Created by: Qt User Interface Compiler version 6.6.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAIN_WINDOW_H
#define UI_MAIN_WINDOW_H

#include <HdrListView.h>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDial>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTreeView>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "GLViewer.h"
#include "MeshListView.h"
#include "ProgressStatusWidget.h"
#include "SceneListView.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionNew_Project;
    QAction *actionImport_image;
    QAction *actionSave_image;
    QAction *actionSave_project;
    QAction *actionExit;
    QAction *actionDocumentation;
    QAction *actionAxomae_version;
    QAction *actionUndo;
    QAction *actionRedo;
    QAction *actionImport_3D_model;
    QAction *actionImport_Environment_Map;
    QWidget *centralwidget;
    QGridLayout *gridLayout_2;
    ProgressStatusWidget *progressBar;
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QGridLayout *gridLayout_4;
    QTabWidget *workspace;
    QWidget *texture;
    QGridLayout *gridLayout_9;
    QGridLayout *gridLayout_5;
    QGridLayout *gridLayout_10;
    QGraphicsView *height_image;
    QGraphicsView *greyscale_image;
    QGraphicsView *dudv_image;
    QGraphicsView *diffuse_image;
    QGraphicsView *normal_image;
    QVBoxLayout *verticalLayout_21;
    QTabWidget *renderMaterials;
    QWidget *functions;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *undo_button;
    QPushButton *redo_button;
    QGroupBox *greyscale_opt;
    QVBoxLayout *verticalLayout_8;
    QRadioButton *use_average;
    QRadioButton *use_luminance;
    QGroupBox *height_opt;
    QVBoxLayout *verticalLayout_3;
    QTabWidget *tabWidget_2;
    QWidget *tab;
    QGridLayout *gridLayout_13;
    QGridLayout *gridLayout_12;
    QRadioButton *use_scharr;
    QRadioButton *use_sobel;
    QRadioButton *use_prewitt;
    QWidget *verticalLayout_16;
    QVBoxLayout *verticalLayout_14;
    QGridLayout *gridLayout_11;
    QVBoxLayout *verticalLayout_13;
    QRadioButton *sharpen_radio;
    QRadioButton *sharpen_masking_radio;
    QDoubleSpinBox *sharpen_float_box;
    QSlider *sharpen_slider;
    QPushButton *sharpen_button;
    QWidget *smoother_grid;
    QGridLayout *gridLayout_7;
    QVBoxLayout *verticalLayout_17;
    QRadioButton *box_blur_radio;
    QRadioButton *gaussian_5_5_radio;
    QRadioButton *gaussian_3_3_radio;
    QSpinBox *smooth_factor;
    QDial *smooth_dial;
    QPushButton *smooth_button;
    QGroupBox *normal_opt;
    QVBoxLayout *verticalLayout_5;
    QRadioButton *use_objectSpace;
    QRadioButton *use_tangentSpace;
    QGroupBox *nmap_factor_opt;
    QVBoxLayout *verticalLayout_4;
    QDoubleSpinBox *factor_nmap;
    QLabel *label;
    QSlider *attenuation_slider_nmap;
    QLabel *label_2;
    QSlider *factor_slider_nmap;
    QGroupBox *dudv_opt;
    QVBoxLayout *verticalLayout_7;
    QPushButton *compute_dudv;
    QGroupBox *nmap_factor_opt_2;
    QVBoxLayout *verticalLayout_6;
    QDoubleSpinBox *factor_dudv;
    QSlider *factor_slider_dudv;
    QWidget *tab_3;
    QHBoxLayout *horizontalLayout_4;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *verticalLayout_25;
    QTabWidget *tabWidget_3;
    QWidget *tab_4;
    QHBoxLayout *horizontalLayout_5;
    QHBoxLayout *horizontalLayout;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_15;
    QGridLayout *gridLayout_3;
    QLabel *label_3;
    QSpinBox *uv_height;
    QLabel *label_4;
    QSpinBox *uv_width;
    QPushButton *bake_texture;
    QGraphicsView *cubemap_render_view;
    QWidget *gl_renderer;
    QGridLayout *gridLayout_6;
    QTabWidget *tabWidget;
    QWidget *PostProTab;
    QVBoxLayout *verticalLayout_10;
    QGroupBox *groupBox_2;
    QFormLayout *formLayout;
    QSlider *gamma_slider;
    QSlider *exposure_slider;
    QLabel *label_5;
    QLabel *label_6;
    QPushButton *reset_camera_button;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_18;
    QRadioButton *set_standard_post_p;
    QRadioButton *set_edge_post_p;
    QRadioButton *set_sharpen_post_p;
    QRadioButton *set_blurr_post_p;
    QWidget *tab_5;
    QVBoxLayout *verticalLayout_9;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_11;
    QRadioButton *rasterize_point_button;
    QRadioButton *rasterize_wireframe_button;
    QRadioButton *rasterize_fill_button;
    QGroupBox *groupBox_5;
    QGridLayout *gridLayout_19;
    QCheckBox *rasterize_display_bbox_checkbox;
    QWidget *tab_2;
    QGridLayout *gridLayout_18;
    QGridLayout *gridLayout_16;
    QTreeView *renderer_lighting_lightlist;
    QTabWidget *tab_renderer_lighting;
    QWidget *tab_9;
    QVBoxLayout *verticalLayout_23;
    QVBoxLayout *verticalLayout_22;
    QGroupBox *groupBox_9;
    QVBoxLayout *verticalLayout_19;
    QTabWidget *tabWidget_4;
    QWidget *tab_7;
    QVBoxLayout *verticalLayout_20;
    QGridLayout *gridLayout_20;
    QSlider *hslider_renderer_lighting_PointLights_colors_red;
    QLabel *label_7;
    QLabel *label_8;
    QSpinBox *spinbox_renderer_lighting_PointLights_colors_green;
    QSlider *hslider_renderer_lighting_PointLights_colors_blue;
    QSlider *hslider_renderer_lighting_PointLights_colors_green;
    QSpinBox *spinbox_renderer_lighting_PointLights_colors_red;
    QSpinBox *spinbox_renderer_lighting_PointLights_colors_blue;
    QLabel *label_9;
    QLabel *label_16;
    QDoubleSpinBox *dspinbox_renderer_lighting_PointLights_intensity;
    QWidget *tab_8;
    QGridLayout *gridLayout_23;
    QGridLayout *gridLayout_22;
    QDoubleSpinBox *dspinbox_renderer_lighting_PointLights_Attenuation_linear;
    QLabel *label_11;
    QDoubleSpinBox *dspinbox_renderer_lighting_PointLights_Attenuation_quadratic;
    QDoubleSpinBox *dspinbox_renderer_lighting_PointLights_Attenuation_constant;
    QLabel *label_10;
    QLabel *label_12;
    QPushButton *button_renderer_lighting_PointLights_add;
    QPushButton *button_renderer_lighting_PointLights_delete;
    QWidget *tab_12;
    QVBoxLayout *verticalLayout_28;
    QVBoxLayout *verticalLayout_27;
    QGroupBox *groupBox_10;
    QVBoxLayout *verticalLayout_24;
    QTabWidget *tabWidget_6;
    QWidget *tab_13;
    QVBoxLayout *verticalLayout_26;
    QGridLayout *gridLayout_21;
    QLabel *label_13;
    QSpinBox *spinbox_renderer_lighting_DirectionalLights_colors_blue;
    QLabel *label_14;
    QSlider *hslider_renderer_lighting_DirectionalLights_colors_blue;
    QSlider *hslider_renderer_lighting_DirectionalLights_colors_green;
    QSpinBox *spinbox_renderer_lighting_DirectionalLight_colors_red;
    QSpinBox *spinbox_renderer_lighting_DirectionalLights_colors_green;
    QSlider *hslider_renderer_lighting_DirectionalLights_colors_red;
    QLabel *label_15;
    QLabel *label_17;
    QDoubleSpinBox *dspinbox_renderer_lighting_DirectionalLights_intensity;
    QPushButton *button_renderer_lighting_DirectionalLights_add;
    QPushButton *button_renderer_lighting_DirectionalLights_delete;
    QWidget *tab_11;
    QGridLayout *gridLayout_28;
    QGridLayout *gridLayout_27;
    QGroupBox *groupBox_11;
    QVBoxLayout *verticalLayout_29;
    QTabWidget *tabWidget_7;
    QWidget *tab_15;
    QVBoxLayout *verticalLayout_30;
    QGridLayout *gridLayout_24;
    QSlider *hslider_renderer_lighting_SpotLights_colors_red;
    QLabel *label_18;
    QLabel *label_19;
    QSpinBox *spinbox_renderer_lighting_SpotLights_colors_green;
    QSlider *hslider_renderer_lighting_SpotLights_colors_blue;
    QSlider *hslider_renderer_lighting_SpotLights_colors_green;
    QSpinBox *spinbox_renderer_lighting_SpotLights_colors_red;
    QSpinBox *spinbox_renderer_lighting_SpotLights_colors_blue;
    QLabel *label_20;
    QLabel *label_21;
    QDoubleSpinBox *dspinbox_renderer_lighting_SpotLights_intensity;
    QWidget *tab_16;
    QGridLayout *gridLayout_25;
    QGridLayout *gridLayout_26;
    QLabel *label_23;
    QDoubleSpinBox *dspinbox_renderer_lighting_SpotLights_angle;
    QDial *dial_renderer_lighting_SpotLights_angle;
    QPushButton *button_renderer_lighting_SpotLights_add;
    QPushButton *button_renderer_lighting_SpotLights_delete;
    QWidget *tab_10;
    QWidget *tab_6;
    QHBoxLayout *horizontalLayout_6;
    QVBoxLayout *verticalLayout_12;
    QGroupBox *groupBox_7;
    QGroupBox *groupBox_6;
    QGridLayout *gridLayout_32;
    GLViewer *renderer_view;
    QVBoxLayout *verticalLayout_31;
    SceneListView *renderer_scene_list;
    EnvmapListDisplay *renderer_envmap_list;
    QWidget *uv_editor;
    QGridLayout *gridLayout_17;
    QGridLayout *gridLayout_14;
    MeshListView *meshes_list;
    QGridLayout *gridLayout_8;
    QPushButton *next_mesh_button;
    QPushButton *previous_mesh_button;
    QGridLayout *gridLayout_15;
    QGraphicsView *uv_projection;
    QWidget *rt_renderer;
    QGridLayout *gridLayout_30;
    QGridLayout *gridLayout_29;
    QCheckBox *use_gpu;
    QMenuBar *menubar;
    QMenu *menuFiles;
    QMenu *menuEdit;
    QMenu *menuTools;
    QMenu *menuHelp;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1528, 934);
        QPalette palette;
        QBrush brush(QColor(255, 255, 255, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::WindowText, brush);
        QBrush brush1(QColor(42, 42, 42, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Button, brush1);
        QBrush brush2(QColor(0, 0, 0, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Light, brush2);
        palette.setBrush(QPalette::Active, QPalette::Midlight, brush2);
        palette.setBrush(QPalette::Active, QPalette::Dark, brush2);
        palette.setBrush(QPalette::Active, QPalette::Mid, brush2);
        palette.setBrush(QPalette::Active, QPalette::Text, brush);
        palette.setBrush(QPalette::Active, QPalette::BrightText, brush);
        palette.setBrush(QPalette::Active, QPalette::ButtonText, brush);
        palette.setBrush(QPalette::Active, QPalette::Base, brush1);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Active, QPalette::Shadow, brush2);
        palette.setBrush(QPalette::Active, QPalette::AlternateBase, brush2);
        QBrush brush3(QColor(255, 255, 220, 255));
        brush3.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::ToolTipBase, brush3);
        palette.setBrush(QPalette::Active, QPalette::ToolTipText, brush2);
        QBrush brush4(QColor(255, 255, 255, 127));
        brush4.setStyle(Qt::SolidPattern);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette::Active, QPalette::PlaceholderText, brush4);
#endif
        QBrush brush5(QColor(211, 218, 227, 255));
        brush5.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::WindowText, brush5);
        palette.setBrush(QPalette::Inactive, QPalette::Button, brush1);
        QBrush brush6(QColor(22, 29, 36, 255));
        brush6.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::Light, brush6);
        QBrush brush7(QColor(32, 43, 54, 255));
        brush7.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::Midlight, brush7);
        QBrush brush8(QColor(86, 114, 144, 255));
        brush8.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::Dark, brush8);
        QBrush brush9(QColor(57, 76, 96, 255));
        brush9.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::Mid, brush9);
        palette.setBrush(QPalette::Inactive, QPalette::Text, brush5);
        palette.setBrush(QPalette::Inactive, QPalette::BrightText, brush);
        palette.setBrush(QPalette::Inactive, QPalette::ButtonText, brush5);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        QBrush brush10(QColor(118, 118, 118, 255));
        brush10.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::Shadow, brush10);
        QBrush brush11(QColor(33, 43, 54, 255));
        brush11.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::AlternateBase, brush11);
        QBrush brush12(QColor(53, 57, 69, 255));
        brush12.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Inactive, QPalette::ToolTipBase, brush12);
        palette.setBrush(QPalette::Inactive, QPalette::ToolTipText, brush5);
        QBrush brush13(QColor(0, 0, 0, 128));
        brush13.setStyle(Qt::SolidPattern);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette::Inactive, QPalette::PlaceholderText, brush13);
#endif
        palette.setBrush(QPalette::Disabled, QPalette::WindowText, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Button, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Light, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Midlight, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Dark, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Mid, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Text, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::BrightText, brush);
        palette.setBrush(QPalette::Disabled, QPalette::ButtonText, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        QBrush brush14(QColor(177, 177, 177, 255));
        brush14.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::Shadow, brush14);
        palette.setBrush(QPalette::Disabled, QPalette::AlternateBase, brush11);
        palette.setBrush(QPalette::Disabled, QPalette::ToolTipBase, brush12);
        palette.setBrush(QPalette::Disabled, QPalette::ToolTipText, brush5);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette::Disabled, QPalette::PlaceholderText, brush13);
#endif
        MainWindow->setPalette(palette);
        MainWindow->setStyleSheet(QString::fromUtf8("QMainWindow,QWidget {\n"
"	background-color :rgb(42, 42, 42);\n"
"	\n"
"}\n"
"\n"
""));
        actionNew_Project = new QAction(MainWindow);
        actionNew_Project->setObjectName("actionNew_Project");
        actionImport_image = new QAction(MainWindow);
        actionImport_image->setObjectName("actionImport_image");
        actionSave_image = new QAction(MainWindow);
        actionSave_image->setObjectName("actionSave_image");
        actionSave_project = new QAction(MainWindow);
        actionSave_project->setObjectName("actionSave_project");
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName("actionExit");
        actionDocumentation = new QAction(MainWindow);
        actionDocumentation->setObjectName("actionDocumentation");
        actionAxomae_version = new QAction(MainWindow);
        actionAxomae_version->setObjectName("actionAxomae_version");
        actionUndo = new QAction(MainWindow);
        actionUndo->setObjectName("actionUndo");
        actionRedo = new QAction(MainWindow);
        actionRedo->setObjectName("actionRedo");
        actionImport_3D_model = new QAction(MainWindow);
        actionImport_3D_model->setObjectName("actionImport_3D_model");
        actionImport_Environment_Map = new QAction(MainWindow);
        actionImport_Environment_Map->setObjectName("actionImport_Environment_Map");
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        gridLayout_2 = new QGridLayout(centralwidget);
        gridLayout_2->setObjectName("gridLayout_2");
        progressBar = new ProgressStatusWidget(centralwidget);
        progressBar->setObjectName("progressBar");
        progressBar->setValue(24);

        gridLayout_2->addWidget(progressBar, 2, 0, 1, 4);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setSizeConstraint(QLayout::SetMaximumSize);
        verticalLayout->setContentsMargins(-1, -1, 0, -1);
        gridLayout = new QGridLayout();
        gridLayout->setObjectName("gridLayout");
        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName("gridLayout_4");
        workspace = new QTabWidget(centralwidget);
        workspace->setObjectName("workspace");
        sizePolicy.setHeightForWidth(workspace->sizePolicy().hasHeightForWidth());
        workspace->setSizePolicy(sizePolicy);
        workspace->setTabShape(QTabWidget::Rounded);
        workspace->setMovable(false);
        workspace->setTabBarAutoHide(false);
        texture = new QWidget();
        texture->setObjectName("texture");
        gridLayout_9 = new QGridLayout(texture);
        gridLayout_9->setObjectName("gridLayout_9");
        gridLayout_5 = new QGridLayout();
        gridLayout_5->setObjectName("gridLayout_5");
        gridLayout_10 = new QGridLayout();
        gridLayout_10->setObjectName("gridLayout_10");
        height_image = new QGraphicsView(texture);
        height_image->setObjectName("height_image");
        height_image->setFrameShape(QFrame::NoFrame);
        height_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContentsOnFirstShow);

        gridLayout_10->addWidget(height_image, 0, 1, 1, 1);

        greyscale_image = new QGraphicsView(texture);
        greyscale_image->setObjectName("greyscale_image");
        greyscale_image->setFrameShape(QFrame::NoFrame);
        greyscale_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContentsOnFirstShow);
        greyscale_image->setDragMode(QGraphicsView::NoDrag);

        gridLayout_10->addWidget(greyscale_image, 0, 0, 1, 1);


        gridLayout_5->addLayout(gridLayout_10, 0, 2, 1, 1);

        dudv_image = new QGraphicsView(texture);
        dudv_image->setObjectName("dudv_image");
        dudv_image->setFrameShape(QFrame::NoFrame);
        dudv_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        dudv_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_5->addWidget(dudv_image, 3, 2, 1, 1);

        diffuse_image = new QGraphicsView(texture);
        diffuse_image->setObjectName("diffuse_image");
        diffuse_image->setFrameShape(QFrame::NoFrame);
        diffuse_image->setFrameShadow(QFrame::Sunken);
        diffuse_image->setLineWidth(1);
        diffuse_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        diffuse_image->setDragMode(QGraphicsView::ScrollHandDrag);
        diffuse_image->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

        gridLayout_5->addWidget(diffuse_image, 0, 1, 1, 1);

        normal_image = new QGraphicsView(texture);
        normal_image->setObjectName("normal_image");
        normal_image->setFrameShape(QFrame::NoFrame);
        normal_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        normal_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_5->addWidget(normal_image, 3, 1, 1, 1);


        gridLayout_9->addLayout(gridLayout_5, 1, 2, 1, 1);

        verticalLayout_21 = new QVBoxLayout();
        verticalLayout_21->setObjectName("verticalLayout_21");

        gridLayout_9->addLayout(verticalLayout_21, 1, 0, 1, 1);

        renderMaterials = new QTabWidget(texture);
        renderMaterials->setObjectName("renderMaterials");
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(renderMaterials->sizePolicy().hasHeightForWidth());
        renderMaterials->setSizePolicy(sizePolicy1);
        renderMaterials->setMinimumSize(QSize(208, 0));
        renderMaterials->setMaximumSize(QSize(208, 813));
        renderMaterials->setStyleSheet(QString::fromUtf8(""));
        renderMaterials->setTabShape(QTabWidget::Triangular);
        renderMaterials->setMovable(true);
        renderMaterials->setTabBarAutoHide(true);
        functions = new QWidget();
        functions->setObjectName("functions");
        verticalLayout_2 = new QVBoxLayout(functions);
        verticalLayout_2->setObjectName("verticalLayout_2");
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        undo_button = new QPushButton(functions);
        undo_button->setObjectName("undo_button");
        QIcon icon(QIcon::fromTheme(QString::fromUtf8("go-previous")));
        undo_button->setIcon(icon);

        horizontalLayout_2->addWidget(undo_button);

        redo_button = new QPushButton(functions);
        redo_button->setObjectName("redo_button");
        QIcon icon1(QIcon::fromTheme(QString::fromUtf8("go-next")));
        redo_button->setIcon(icon1);

        horizontalLayout_2->addWidget(redo_button);


        verticalLayout_2->addLayout(horizontalLayout_2);

        greyscale_opt = new QGroupBox(functions);
        greyscale_opt->setObjectName("greyscale_opt");
        greyscale_opt->setEnabled(true);
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(greyscale_opt->sizePolicy().hasHeightForWidth());
        greyscale_opt->setSizePolicy(sizePolicy2);
        greyscale_opt->setFlat(false);
        greyscale_opt->setCheckable(false);
        verticalLayout_8 = new QVBoxLayout(greyscale_opt);
        verticalLayout_8->setObjectName("verticalLayout_8");
        use_average = new QRadioButton(greyscale_opt);
        use_average->setObjectName("use_average");
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(use_average->sizePolicy().hasHeightForWidth());
        use_average->setSizePolicy(sizePolicy3);
        QFont font;
        font.setFamilies({QString::fromUtf8("MS Shell Dlg 2")});
        use_average->setFont(font);
        use_average->setChecked(true);

        verticalLayout_8->addWidget(use_average);

        use_luminance = new QRadioButton(greyscale_opt);
        use_luminance->setObjectName("use_luminance");
        sizePolicy3.setHeightForWidth(use_luminance->sizePolicy().hasHeightForWidth());
        use_luminance->setSizePolicy(sizePolicy3);

        verticalLayout_8->addWidget(use_luminance);


        verticalLayout_2->addWidget(greyscale_opt);

        height_opt = new QGroupBox(functions);
        height_opt->setObjectName("height_opt");
        verticalLayout_3 = new QVBoxLayout(height_opt);
        verticalLayout_3->setObjectName("verticalLayout_3");
        tabWidget_2 = new QTabWidget(height_opt);
        tabWidget_2->setObjectName("tabWidget_2");
        tabWidget_2->setTabPosition(QTabWidget::North);
        tabWidget_2->setTabShape(QTabWidget::Triangular);
        tabWidget_2->setTabsClosable(false);
        tabWidget_2->setTabBarAutoHide(true);
        tab = new QWidget();
        tab->setObjectName("tab");
        gridLayout_13 = new QGridLayout(tab);
        gridLayout_13->setObjectName("gridLayout_13");
        gridLayout_12 = new QGridLayout();
        gridLayout_12->setObjectName("gridLayout_12");
        use_scharr = new QRadioButton(tab);
        use_scharr->setObjectName("use_scharr");

        gridLayout_12->addWidget(use_scharr, 2, 0, 1, 1);

        use_sobel = new QRadioButton(tab);
        use_sobel->setObjectName("use_sobel");
        use_sobel->setChecked(true);

        gridLayout_12->addWidget(use_sobel, 0, 0, 1, 1);

        use_prewitt = new QRadioButton(tab);
        use_prewitt->setObjectName("use_prewitt");

        gridLayout_12->addWidget(use_prewitt, 1, 0, 1, 1);


        gridLayout_13->addLayout(gridLayout_12, 0, 0, 1, 1);

        tabWidget_2->addTab(tab, QString());
        verticalLayout_16 = new QWidget();
        verticalLayout_16->setObjectName("verticalLayout_16");
        verticalLayout_14 = new QVBoxLayout(verticalLayout_16);
        verticalLayout_14->setObjectName("verticalLayout_14");
        gridLayout_11 = new QGridLayout();
        gridLayout_11->setObjectName("gridLayout_11");
        verticalLayout_13 = new QVBoxLayout();
        verticalLayout_13->setObjectName("verticalLayout_13");
        sharpen_radio = new QRadioButton(verticalLayout_16);
        sharpen_radio->setObjectName("sharpen_radio");
        sharpen_radio->setChecked(true);

        verticalLayout_13->addWidget(sharpen_radio);

        sharpen_masking_radio = new QRadioButton(verticalLayout_16);
        sharpen_masking_radio->setObjectName("sharpen_masking_radio");

        verticalLayout_13->addWidget(sharpen_masking_radio);

        sharpen_float_box = new QDoubleSpinBox(verticalLayout_16);
        sharpen_float_box->setObjectName("sharpen_float_box");

        verticalLayout_13->addWidget(sharpen_float_box);

        sharpen_slider = new QSlider(verticalLayout_16);
        sharpen_slider->setObjectName("sharpen_slider");
        sharpen_slider->setMinimum(1);
        sharpen_slider->setMaximum(10);
        sharpen_slider->setOrientation(Qt::Horizontal);

        verticalLayout_13->addWidget(sharpen_slider);

        sharpen_button = new QPushButton(verticalLayout_16);
        sharpen_button->setObjectName("sharpen_button");

        verticalLayout_13->addWidget(sharpen_button);


        gridLayout_11->addLayout(verticalLayout_13, 0, 0, 1, 1);


        verticalLayout_14->addLayout(gridLayout_11);

        tabWidget_2->addTab(verticalLayout_16, QString());
        smoother_grid = new QWidget();
        smoother_grid->setObjectName("smoother_grid");
        gridLayout_7 = new QGridLayout(smoother_grid);
        gridLayout_7->setObjectName("gridLayout_7");
        verticalLayout_17 = new QVBoxLayout();
        verticalLayout_17->setObjectName("verticalLayout_17");
        verticalLayout_17->setSizeConstraint(QLayout::SetNoConstraint);
        box_blur_radio = new QRadioButton(smoother_grid);
        box_blur_radio->setObjectName("box_blur_radio");
        box_blur_radio->setChecked(true);

        verticalLayout_17->addWidget(box_blur_radio);

        gaussian_5_5_radio = new QRadioButton(smoother_grid);
        gaussian_5_5_radio->setObjectName("gaussian_5_5_radio");

        verticalLayout_17->addWidget(gaussian_5_5_radio);

        gaussian_3_3_radio = new QRadioButton(smoother_grid);
        gaussian_3_3_radio->setObjectName("gaussian_3_3_radio");

        verticalLayout_17->addWidget(gaussian_3_3_radio);

        smooth_factor = new QSpinBox(smoother_grid);
        smooth_factor->setObjectName("smooth_factor");

        verticalLayout_17->addWidget(smooth_factor);

        smooth_dial = new QDial(smoother_grid);
        smooth_dial->setObjectName("smooth_dial");
        smooth_dial->setMaximum(10);
        smooth_dial->setOrientation(Qt::Vertical);
        smooth_dial->setInvertedAppearance(false);

        verticalLayout_17->addWidget(smooth_dial);

        smooth_button = new QPushButton(smoother_grid);
        smooth_button->setObjectName("smooth_button");

        verticalLayout_17->addWidget(smooth_button);


        gridLayout_7->addLayout(verticalLayout_17, 0, 0, 1, 1);

        tabWidget_2->addTab(smoother_grid, QString());

        verticalLayout_3->addWidget(tabWidget_2);


        verticalLayout_2->addWidget(height_opt);

        normal_opt = new QGroupBox(functions);
        normal_opt->setObjectName("normal_opt");
        verticalLayout_5 = new QVBoxLayout(normal_opt);
        verticalLayout_5->setObjectName("verticalLayout_5");
        use_objectSpace = new QRadioButton(normal_opt);
        use_objectSpace->setObjectName("use_objectSpace");
        use_objectSpace->setChecked(true);

        verticalLayout_5->addWidget(use_objectSpace);

        use_tangentSpace = new QRadioButton(normal_opt);
        use_tangentSpace->setObjectName("use_tangentSpace");

        verticalLayout_5->addWidget(use_tangentSpace);

        nmap_factor_opt = new QGroupBox(normal_opt);
        nmap_factor_opt->setObjectName("nmap_factor_opt");
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(nmap_factor_opt->sizePolicy().hasHeightForWidth());
        nmap_factor_opt->setSizePolicy(sizePolicy4);
        verticalLayout_4 = new QVBoxLayout(nmap_factor_opt);
        verticalLayout_4->setObjectName("verticalLayout_4");
        factor_nmap = new QDoubleSpinBox(nmap_factor_opt);
        factor_nmap->setObjectName("factor_nmap");

        verticalLayout_4->addWidget(factor_nmap);

        label = new QLabel(nmap_factor_opt);
        label->setObjectName("label");

        verticalLayout_4->addWidget(label);

        attenuation_slider_nmap = new QSlider(nmap_factor_opt);
        attenuation_slider_nmap->setObjectName("attenuation_slider_nmap");
        attenuation_slider_nmap->setMinimum(1);
        attenuation_slider_nmap->setMaximum(10);
        attenuation_slider_nmap->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(attenuation_slider_nmap);

        label_2 = new QLabel(nmap_factor_opt);
        label_2->setObjectName("label_2");

        verticalLayout_4->addWidget(label_2);

        factor_slider_nmap = new QSlider(nmap_factor_opt);
        factor_slider_nmap->setObjectName("factor_slider_nmap");
        factor_slider_nmap->setStyleSheet(QString::fromUtf8("QSlider {\n"
"qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 255, 255, 255), stop:0.1 rgba(255, 255, 255, 255), stop:0.2 rgba(255, 176, 176, 167), stop:0.3 rgba(255, 151, 151, 92), stop:0.4 rgba(255, 125, 125, 51), stop:0.5 rgba(255, 76, 76, 205), stop:0.52 rgba(255, 76, 76, 205), stop:0.6 rgba(255, 180, 180, 84), stop:1 rgba(255, 255, 255, 0))\n"
"\n"
"}"));
        factor_slider_nmap->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(factor_slider_nmap);


        verticalLayout_5->addWidget(nmap_factor_opt);


        verticalLayout_2->addWidget(normal_opt);

        dudv_opt = new QGroupBox(functions);
        dudv_opt->setObjectName("dudv_opt");
        verticalLayout_7 = new QVBoxLayout(dudv_opt);
        verticalLayout_7->setObjectName("verticalLayout_7");
        compute_dudv = new QPushButton(dudv_opt);
        compute_dudv->setObjectName("compute_dudv");

        verticalLayout_7->addWidget(compute_dudv);

        nmap_factor_opt_2 = new QGroupBox(dudv_opt);
        nmap_factor_opt_2->setObjectName("nmap_factor_opt_2");
        sizePolicy4.setHeightForWidth(nmap_factor_opt_2->sizePolicy().hasHeightForWidth());
        nmap_factor_opt_2->setSizePolicy(sizePolicy4);
        verticalLayout_6 = new QVBoxLayout(nmap_factor_opt_2);
        verticalLayout_6->setObjectName("verticalLayout_6");
        factor_dudv = new QDoubleSpinBox(nmap_factor_opt_2);
        factor_dudv->setObjectName("factor_dudv");

        verticalLayout_6->addWidget(factor_dudv);

        factor_slider_dudv = new QSlider(nmap_factor_opt_2);
        factor_slider_dudv->setObjectName("factor_slider_dudv");
        factor_slider_dudv->setStyleSheet(QString::fromUtf8("QSlider {\n"
"qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 255, 255, 255), stop:0.1 rgba(255, 255, 255, 255), stop:0.2 rgba(255, 176, 176, 167), stop:0.3 rgba(255, 151, 151, 92), stop:0.4 rgba(255, 125, 125, 51), stop:0.5 rgba(255, 76, 76, 205), stop:0.52 rgba(255, 76, 76, 205), stop:0.6 rgba(255, 180, 180, 84), stop:1 rgba(255, 255, 255, 0))\n"
"\n"
"}"));
        factor_slider_dudv->setOrientation(Qt::Horizontal);

        verticalLayout_6->addWidget(factor_slider_dudv);


        verticalLayout_7->addWidget(nmap_factor_opt_2);


        verticalLayout_2->addWidget(dudv_opt);

        renderMaterials->addTab(functions, QString());

        gridLayout_9->addWidget(renderMaterials, 1, 1, 1, 1);

        workspace->addTab(texture, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName("tab_3");
        horizontalLayout_4 = new QHBoxLayout(tab_3);
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        verticalLayout_25 = new QVBoxLayout();
        verticalLayout_25->setObjectName("verticalLayout_25");
        verticalLayout_25->setSizeConstraint(QLayout::SetMinimumSize);
        tabWidget_3 = new QTabWidget(tab_3);
        tabWidget_3->setObjectName("tabWidget_3");
        QSizePolicy sizePolicy5(QSizePolicy::Maximum, QSizePolicy::Expanding);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(tabWidget_3->sizePolicy().hasHeightForWidth());
        tabWidget_3->setSizePolicy(sizePolicy5);
        tab_4 = new QWidget();
        tab_4->setObjectName("tab_4");
        horizontalLayout_5 = new QHBoxLayout(tab_4);
        horizontalLayout_5->setObjectName("horizontalLayout_5");
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        groupBox = new QGroupBox(tab_4);
        groupBox->setObjectName("groupBox");
        QSizePolicy sizePolicy6(QSizePolicy::Maximum, QSizePolicy::Maximum);
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy6);
        verticalLayout_15 = new QVBoxLayout(groupBox);
        verticalLayout_15->setObjectName("verticalLayout_15");
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName("gridLayout_3");
        label_3 = new QLabel(groupBox);
        label_3->setObjectName("label_3");

        gridLayout_3->addWidget(label_3, 1, 0, 1, 1);

        uv_height = new QSpinBox(groupBox);
        uv_height->setObjectName("uv_height");
        uv_height->setMinimum(50);
        uv_height->setMaximum(8096);
        uv_height->setValue(500);

        gridLayout_3->addWidget(uv_height, 1, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName("label_4");

        gridLayout_3->addWidget(label_4, 0, 0, 1, 1);

        uv_width = new QSpinBox(groupBox);
        uv_width->setObjectName("uv_width");
        uv_width->setMinimum(50);
        uv_width->setMaximum(8096);
        uv_width->setValue(500);

        gridLayout_3->addWidget(uv_width, 0, 1, 1, 1);


        verticalLayout_15->addLayout(gridLayout_3);

        bake_texture = new QPushButton(groupBox);
        bake_texture->setObjectName("bake_texture");

        verticalLayout_15->addWidget(bake_texture);


        horizontalLayout->addWidget(groupBox);


        horizontalLayout_5->addLayout(horizontalLayout);

        tabWidget_3->addTab(tab_4, QString());

        verticalLayout_25->addWidget(tabWidget_3);


        horizontalLayout_3->addLayout(verticalLayout_25);

        cubemap_render_view = new QGraphicsView(tab_3);
        cubemap_render_view->setObjectName("cubemap_render_view");
        sizePolicy.setHeightForWidth(cubemap_render_view->sizePolicy().hasHeightForWidth());
        cubemap_render_view->setSizePolicy(sizePolicy);

        horizontalLayout_3->addWidget(cubemap_render_view);


        horizontalLayout_4->addLayout(horizontalLayout_3);

        workspace->addTab(tab_3, QString());
        gl_renderer = new QWidget();
        gl_renderer->setObjectName("gl_renderer");
        sizePolicy2.setHeightForWidth(gl_renderer->sizePolicy().hasHeightForWidth());
        gl_renderer->setSizePolicy(sizePolicy2);
        gridLayout_6 = new QGridLayout(gl_renderer);
        gridLayout_6->setObjectName("gridLayout_6");
        tabWidget = new QTabWidget(gl_renderer);
        tabWidget->setObjectName("tabWidget");
        QSizePolicy sizePolicy7(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy7);
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setElideMode(Qt::ElideNone);
        tabWidget->setTabsClosable(false);
        PostProTab = new QWidget();
        PostProTab->setObjectName("PostProTab");
        PostProTab->setEnabled(true);
        PostProTab->setAutoFillBackground(false);
        verticalLayout_10 = new QVBoxLayout(PostProTab);
        verticalLayout_10->setObjectName("verticalLayout_10");
        groupBox_2 = new QGroupBox(PostProTab);
        groupBox_2->setObjectName("groupBox_2");
        QSizePolicy sizePolicy8(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(0);
        sizePolicy8.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy8);
        groupBox_2->setFocusPolicy(Qt::NoFocus);
        groupBox_2->setAutoFillBackground(false);
        groupBox_2->setFlat(false);
        formLayout = new QFormLayout(groupBox_2);
        formLayout->setObjectName("formLayout");
        gamma_slider = new QSlider(groupBox_2);
        gamma_slider->setObjectName("gamma_slider");
        gamma_slider->setMaximum(200);
        gamma_slider->setOrientation(Qt::Horizontal);
        gamma_slider->setTickPosition(QSlider::TicksBelow);

        formLayout->setWidget(1, QFormLayout::LabelRole, gamma_slider);

        exposure_slider = new QSlider(groupBox_2);
        exposure_slider->setObjectName("exposure_slider");
        exposure_slider->setMaximum(200);
        exposure_slider->setOrientation(Qt::Horizontal);
        exposure_slider->setTickPosition(QSlider::TicksBelow);

        formLayout->setWidget(3, QFormLayout::LabelRole, exposure_slider);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName("label_5");

        formLayout->setWidget(0, QFormLayout::LabelRole, label_5);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName("label_6");

        formLayout->setWidget(2, QFormLayout::LabelRole, label_6);

        reset_camera_button = new QPushButton(groupBox_2);
        reset_camera_button->setObjectName("reset_camera_button");

        formLayout->setWidget(4, QFormLayout::LabelRole, reset_camera_button);

        groupBox_4 = new QGroupBox(groupBox_2);
        groupBox_4->setObjectName("groupBox_4");
        sizePolicy7.setHeightForWidth(groupBox_4->sizePolicy().hasHeightForWidth());
        groupBox_4->setSizePolicy(sizePolicy7);
        verticalLayout_18 = new QVBoxLayout(groupBox_4);
        verticalLayout_18->setObjectName("verticalLayout_18");
        set_standard_post_p = new QRadioButton(groupBox_4);
        set_standard_post_p->setObjectName("set_standard_post_p");

        verticalLayout_18->addWidget(set_standard_post_p);

        set_edge_post_p = new QRadioButton(groupBox_4);
        set_edge_post_p->setObjectName("set_edge_post_p");

        verticalLayout_18->addWidget(set_edge_post_p);

        set_sharpen_post_p = new QRadioButton(groupBox_4);
        set_sharpen_post_p->setObjectName("set_sharpen_post_p");

        verticalLayout_18->addWidget(set_sharpen_post_p);

        set_blurr_post_p = new QRadioButton(groupBox_4);
        set_blurr_post_p->setObjectName("set_blurr_post_p");

        verticalLayout_18->addWidget(set_blurr_post_p);


        formLayout->setWidget(5, QFormLayout::LabelRole, groupBox_4);


        verticalLayout_10->addWidget(groupBox_2);

        tabWidget->addTab(PostProTab, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName("tab_5");
        verticalLayout_9 = new QVBoxLayout(tab_5);
        verticalLayout_9->setObjectName("verticalLayout_9");
        groupBox_3 = new QGroupBox(tab_5);
        groupBox_3->setObjectName("groupBox_3");
        QSizePolicy sizePolicy9(QSizePolicy::Minimum, QSizePolicy::Maximum);
        sizePolicy9.setHorizontalStretch(0);
        sizePolicy9.setVerticalStretch(0);
        sizePolicy9.setHeightForWidth(groupBox_3->sizePolicy().hasHeightForWidth());
        groupBox_3->setSizePolicy(sizePolicy9);
        verticalLayout_11 = new QVBoxLayout(groupBox_3);
        verticalLayout_11->setObjectName("verticalLayout_11");
        rasterize_point_button = new QRadioButton(groupBox_3);
        rasterize_point_button->setObjectName("rasterize_point_button");

        verticalLayout_11->addWidget(rasterize_point_button);

        rasterize_wireframe_button = new QRadioButton(groupBox_3);
        rasterize_wireframe_button->setObjectName("rasterize_wireframe_button");

        verticalLayout_11->addWidget(rasterize_wireframe_button);

        rasterize_fill_button = new QRadioButton(groupBox_3);
        rasterize_fill_button->setObjectName("rasterize_fill_button");

        verticalLayout_11->addWidget(rasterize_fill_button);

        groupBox_5 = new QGroupBox(groupBox_3);
        groupBox_5->setObjectName("groupBox_5");
        sizePolicy8.setHeightForWidth(groupBox_5->sizePolicy().hasHeightForWidth());
        groupBox_5->setSizePolicy(sizePolicy8);
        gridLayout_19 = new QGridLayout(groupBox_5);
        gridLayout_19->setObjectName("gridLayout_19");
        rasterize_display_bbox_checkbox = new QCheckBox(groupBox_5);
        rasterize_display_bbox_checkbox->setObjectName("rasterize_display_bbox_checkbox");

        gridLayout_19->addWidget(rasterize_display_bbox_checkbox, 1, 0, 1, 2);


        verticalLayout_11->addWidget(groupBox_5);


        verticalLayout_9->addWidget(groupBox_3);

        tabWidget->addTab(tab_5, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName("tab_2");
        gridLayout_18 = new QGridLayout(tab_2);
        gridLayout_18->setObjectName("gridLayout_18");
        gridLayout_16 = new QGridLayout();
        gridLayout_16->setObjectName("gridLayout_16");
        renderer_lighting_lightlist = new QTreeView(tab_2);
        renderer_lighting_lightlist->setObjectName("renderer_lighting_lightlist");

        gridLayout_16->addWidget(renderer_lighting_lightlist, 1, 2, 1, 1);

        tab_renderer_lighting = new QTabWidget(tab_2);
        tab_renderer_lighting->setObjectName("tab_renderer_lighting");
        QSizePolicy sizePolicy10(QSizePolicy::Expanding, QSizePolicy::Maximum);
        sizePolicy10.setHorizontalStretch(0);
        sizePolicy10.setVerticalStretch(0);
        sizePolicy10.setHeightForWidth(tab_renderer_lighting->sizePolicy().hasHeightForWidth());
        tab_renderer_lighting->setSizePolicy(sizePolicy10);
        tab_renderer_lighting->setTabPosition(QTabWidget::North);
        tab_renderer_lighting->setTabShape(QTabWidget::Rounded);
        tab_renderer_lighting->setElideMode(Qt::ElideNone);
        tab_renderer_lighting->setDocumentMode(true);
        tab_renderer_lighting->setTabBarAutoHide(false);
        tab_9 = new QWidget();
        tab_9->setObjectName("tab_9");
        verticalLayout_23 = new QVBoxLayout(tab_9);
        verticalLayout_23->setObjectName("verticalLayout_23");
        verticalLayout_22 = new QVBoxLayout();
        verticalLayout_22->setObjectName("verticalLayout_22");
        groupBox_9 = new QGroupBox(tab_9);
        groupBox_9->setObjectName("groupBox_9");
        sizePolicy10.setHeightForWidth(groupBox_9->sizePolicy().hasHeightForWidth());
        groupBox_9->setSizePolicy(sizePolicy10);
        groupBox_9->setFocusPolicy(Qt::NoFocus);
        groupBox_9->setAcceptDrops(false);
        groupBox_9->setAutoFillBackground(false);
        groupBox_9->setFlat(false);
        groupBox_9->setCheckable(false);
        groupBox_9->setChecked(false);
        verticalLayout_19 = new QVBoxLayout(groupBox_9);
        verticalLayout_19->setObjectName("verticalLayout_19");
        tabWidget_4 = new QTabWidget(groupBox_9);
        tabWidget_4->setObjectName("tabWidget_4");
        sizePolicy6.setHeightForWidth(tabWidget_4->sizePolicy().hasHeightForWidth());
        tabWidget_4->setSizePolicy(sizePolicy6);
        tabWidget_4->setDocumentMode(false);
        tabWidget_4->setTabsClosable(false);
        tabWidget_4->setTabBarAutoHide(false);
        tab_7 = new QWidget();
        tab_7->setObjectName("tab_7");
        verticalLayout_20 = new QVBoxLayout(tab_7);
        verticalLayout_20->setObjectName("verticalLayout_20");
        gridLayout_20 = new QGridLayout();
        gridLayout_20->setObjectName("gridLayout_20");
        hslider_renderer_lighting_PointLights_colors_red = new QSlider(tab_7);
        hslider_renderer_lighting_PointLights_colors_red->setObjectName("hslider_renderer_lighting_PointLights_colors_red");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_PointLights_colors_red->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_PointLights_colors_red->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_PointLights_colors_red->setMaximum(255);
        hslider_renderer_lighting_PointLights_colors_red->setValue(0);
        hslider_renderer_lighting_PointLights_colors_red->setOrientation(Qt::Horizontal);

        gridLayout_20->addWidget(hslider_renderer_lighting_PointLights_colors_red, 4, 0, 1, 1);

        label_7 = new QLabel(tab_7);
        label_7->setObjectName("label_7");

        gridLayout_20->addWidget(label_7, 3, 0, 1, 1);

        label_8 = new QLabel(tab_7);
        label_8->setObjectName("label_8");

        gridLayout_20->addWidget(label_8, 5, 0, 1, 1);

        spinbox_renderer_lighting_PointLights_colors_green = new QSpinBox(tab_7);
        spinbox_renderer_lighting_PointLights_colors_green->setObjectName("spinbox_renderer_lighting_PointLights_colors_green");
        spinbox_renderer_lighting_PointLights_colors_green->setReadOnly(true);
        spinbox_renderer_lighting_PointLights_colors_green->setMaximum(255);

        gridLayout_20->addWidget(spinbox_renderer_lighting_PointLights_colors_green, 6, 1, 1, 1);

        hslider_renderer_lighting_PointLights_colors_blue = new QSlider(tab_7);
        hslider_renderer_lighting_PointLights_colors_blue->setObjectName("hslider_renderer_lighting_PointLights_colors_blue");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_PointLights_colors_blue->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_PointLights_colors_blue->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_PointLights_colors_blue->setMaximum(255);
        hslider_renderer_lighting_PointLights_colors_blue->setOrientation(Qt::Horizontal);

        gridLayout_20->addWidget(hslider_renderer_lighting_PointLights_colors_blue, 8, 0, 1, 1);

        hslider_renderer_lighting_PointLights_colors_green = new QSlider(tab_7);
        hslider_renderer_lighting_PointLights_colors_green->setObjectName("hslider_renderer_lighting_PointLights_colors_green");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_PointLights_colors_green->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_PointLights_colors_green->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_PointLights_colors_green->setMaximum(255);
        hslider_renderer_lighting_PointLights_colors_green->setOrientation(Qt::Horizontal);

        gridLayout_20->addWidget(hslider_renderer_lighting_PointLights_colors_green, 6, 0, 1, 1);

        spinbox_renderer_lighting_PointLights_colors_red = new QSpinBox(tab_7);
        spinbox_renderer_lighting_PointLights_colors_red->setObjectName("spinbox_renderer_lighting_PointLights_colors_red");
        spinbox_renderer_lighting_PointLights_colors_red->setToolTipDuration(-1);
        spinbox_renderer_lighting_PointLights_colors_red->setReadOnly(false);
        spinbox_renderer_lighting_PointLights_colors_red->setMaximum(255);

        gridLayout_20->addWidget(spinbox_renderer_lighting_PointLights_colors_red, 4, 1, 1, 1);

        spinbox_renderer_lighting_PointLights_colors_blue = new QSpinBox(tab_7);
        spinbox_renderer_lighting_PointLights_colors_blue->setObjectName("spinbox_renderer_lighting_PointLights_colors_blue");
        spinbox_renderer_lighting_PointLights_colors_blue->setReadOnly(true);
        spinbox_renderer_lighting_PointLights_colors_blue->setMaximum(255);

        gridLayout_20->addWidget(spinbox_renderer_lighting_PointLights_colors_blue, 8, 1, 1, 1);

        label_9 = new QLabel(tab_7);
        label_9->setObjectName("label_9");

        gridLayout_20->addWidget(label_9, 7, 0, 1, 1);


        verticalLayout_20->addLayout(gridLayout_20);

        label_16 = new QLabel(tab_7);
        label_16->setObjectName("label_16");

        verticalLayout_20->addWidget(label_16);

        dspinbox_renderer_lighting_PointLights_intensity = new QDoubleSpinBox(tab_7);
        dspinbox_renderer_lighting_PointLights_intensity->setObjectName("dspinbox_renderer_lighting_PointLights_intensity");
        dspinbox_renderer_lighting_PointLights_intensity->setDecimals(3);
        dspinbox_renderer_lighting_PointLights_intensity->setMaximum(10000.000000000000000);

        verticalLayout_20->addWidget(dspinbox_renderer_lighting_PointLights_intensity);

        tabWidget_4->addTab(tab_7, QString());
        tab_8 = new QWidget();
        tab_8->setObjectName("tab_8");
        gridLayout_23 = new QGridLayout(tab_8);
        gridLayout_23->setObjectName("gridLayout_23");
        gridLayout_22 = new QGridLayout();
        gridLayout_22->setObjectName("gridLayout_22");
        dspinbox_renderer_lighting_PointLights_Attenuation_linear = new QDoubleSpinBox(tab_8);
        dspinbox_renderer_lighting_PointLights_Attenuation_linear->setObjectName("dspinbox_renderer_lighting_PointLights_Attenuation_linear");
        dspinbox_renderer_lighting_PointLights_Attenuation_linear->setDecimals(4);
        dspinbox_renderer_lighting_PointLights_Attenuation_linear->setMaximum(10000000000000.000000000000000);

        gridLayout_22->addWidget(dspinbox_renderer_lighting_PointLights_Attenuation_linear, 3, 0, 1, 1);

        label_11 = new QLabel(tab_8);
        label_11->setObjectName("label_11");

        gridLayout_22->addWidget(label_11, 2, 0, 1, 1);

        dspinbox_renderer_lighting_PointLights_Attenuation_quadratic = new QDoubleSpinBox(tab_8);
        dspinbox_renderer_lighting_PointLights_Attenuation_quadratic->setObjectName("dspinbox_renderer_lighting_PointLights_Attenuation_quadratic");
        dspinbox_renderer_lighting_PointLights_Attenuation_quadratic->setDecimals(4);
        dspinbox_renderer_lighting_PointLights_Attenuation_quadratic->setMaximum(10000000000000.000000000000000);

        gridLayout_22->addWidget(dspinbox_renderer_lighting_PointLights_Attenuation_quadratic, 5, 0, 1, 1);

        dspinbox_renderer_lighting_PointLights_Attenuation_constant = new QDoubleSpinBox(tab_8);
        dspinbox_renderer_lighting_PointLights_Attenuation_constant->setObjectName("dspinbox_renderer_lighting_PointLights_Attenuation_constant");
        dspinbox_renderer_lighting_PointLights_Attenuation_constant->setDecimals(4);
        dspinbox_renderer_lighting_PointLights_Attenuation_constant->setMaximum(10000000000000.000000000000000);

        gridLayout_22->addWidget(dspinbox_renderer_lighting_PointLights_Attenuation_constant, 1, 0, 1, 1);

        label_10 = new QLabel(tab_8);
        label_10->setObjectName("label_10");

        gridLayout_22->addWidget(label_10, 0, 0, 1, 1);

        label_12 = new QLabel(tab_8);
        label_12->setObjectName("label_12");

        gridLayout_22->addWidget(label_12, 4, 0, 1, 1);


        gridLayout_23->addLayout(gridLayout_22, 0, 0, 1, 1);

        tabWidget_4->addTab(tab_8, QString());

        verticalLayout_19->addWidget(tabWidget_4);

        button_renderer_lighting_PointLights_add = new QPushButton(groupBox_9);
        button_renderer_lighting_PointLights_add->setObjectName("button_renderer_lighting_PointLights_add");

        verticalLayout_19->addWidget(button_renderer_lighting_PointLights_add);

        button_renderer_lighting_PointLights_delete = new QPushButton(groupBox_9);
        button_renderer_lighting_PointLights_delete->setObjectName("button_renderer_lighting_PointLights_delete");

        verticalLayout_19->addWidget(button_renderer_lighting_PointLights_delete);


        verticalLayout_22->addWidget(groupBox_9);


        verticalLayout_23->addLayout(verticalLayout_22);

        tab_renderer_lighting->addTab(tab_9, QString());
        tab_12 = new QWidget();
        tab_12->setObjectName("tab_12");
        verticalLayout_28 = new QVBoxLayout(tab_12);
        verticalLayout_28->setObjectName("verticalLayout_28");
        verticalLayout_27 = new QVBoxLayout();
        verticalLayout_27->setObjectName("verticalLayout_27");
        groupBox_10 = new QGroupBox(tab_12);
        groupBox_10->setObjectName("groupBox_10");
        sizePolicy10.setHeightForWidth(groupBox_10->sizePolicy().hasHeightForWidth());
        groupBox_10->setSizePolicy(sizePolicy10);
        groupBox_10->setFocusPolicy(Qt::NoFocus);
        groupBox_10->setAcceptDrops(false);
        groupBox_10->setAutoFillBackground(false);
        groupBox_10->setFlat(false);
        groupBox_10->setCheckable(false);
        groupBox_10->setChecked(false);
        verticalLayout_24 = new QVBoxLayout(groupBox_10);
        verticalLayout_24->setObjectName("verticalLayout_24");
        tabWidget_6 = new QTabWidget(groupBox_10);
        tabWidget_6->setObjectName("tabWidget_6");
        sizePolicy6.setHeightForWidth(tabWidget_6->sizePolicy().hasHeightForWidth());
        tabWidget_6->setSizePolicy(sizePolicy6);
        tabWidget_6->setDocumentMode(false);
        tabWidget_6->setTabsClosable(false);
        tabWidget_6->setTabBarAutoHide(false);
        tab_13 = new QWidget();
        tab_13->setObjectName("tab_13");
        verticalLayout_26 = new QVBoxLayout(tab_13);
        verticalLayout_26->setObjectName("verticalLayout_26");
        gridLayout_21 = new QGridLayout();
        gridLayout_21->setObjectName("gridLayout_21");
        label_13 = new QLabel(tab_13);
        label_13->setObjectName("label_13");

        gridLayout_21->addWidget(label_13, 3, 0, 1, 1);

        spinbox_renderer_lighting_DirectionalLights_colors_blue = new QSpinBox(tab_13);
        spinbox_renderer_lighting_DirectionalLights_colors_blue->setObjectName("spinbox_renderer_lighting_DirectionalLights_colors_blue");
        spinbox_renderer_lighting_DirectionalLights_colors_blue->setReadOnly(true);
        spinbox_renderer_lighting_DirectionalLights_colors_blue->setMaximum(255);

        gridLayout_21->addWidget(spinbox_renderer_lighting_DirectionalLights_colors_blue, 8, 1, 1, 1);

        label_14 = new QLabel(tab_13);
        label_14->setObjectName("label_14");

        gridLayout_21->addWidget(label_14, 5, 0, 1, 1);

        hslider_renderer_lighting_DirectionalLights_colors_blue = new QSlider(tab_13);
        hslider_renderer_lighting_DirectionalLights_colors_blue->setObjectName("hslider_renderer_lighting_DirectionalLights_colors_blue");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_DirectionalLights_colors_blue->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_DirectionalLights_colors_blue->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_DirectionalLights_colors_blue->setMaximum(255);
        hslider_renderer_lighting_DirectionalLights_colors_blue->setOrientation(Qt::Horizontal);

        gridLayout_21->addWidget(hslider_renderer_lighting_DirectionalLights_colors_blue, 8, 0, 1, 1);

        hslider_renderer_lighting_DirectionalLights_colors_green = new QSlider(tab_13);
        hslider_renderer_lighting_DirectionalLights_colors_green->setObjectName("hslider_renderer_lighting_DirectionalLights_colors_green");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_DirectionalLights_colors_green->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_DirectionalLights_colors_green->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_DirectionalLights_colors_green->setMaximum(255);
        hslider_renderer_lighting_DirectionalLights_colors_green->setOrientation(Qt::Horizontal);

        gridLayout_21->addWidget(hslider_renderer_lighting_DirectionalLights_colors_green, 6, 0, 1, 1);

        spinbox_renderer_lighting_DirectionalLight_colors_red = new QSpinBox(tab_13);
        spinbox_renderer_lighting_DirectionalLight_colors_red->setObjectName("spinbox_renderer_lighting_DirectionalLight_colors_red");
        spinbox_renderer_lighting_DirectionalLight_colors_red->setReadOnly(true);
        spinbox_renderer_lighting_DirectionalLight_colors_red->setMaximum(255);

        gridLayout_21->addWidget(spinbox_renderer_lighting_DirectionalLight_colors_red, 4, 1, 1, 1);

        spinbox_renderer_lighting_DirectionalLights_colors_green = new QSpinBox(tab_13);
        spinbox_renderer_lighting_DirectionalLights_colors_green->setObjectName("spinbox_renderer_lighting_DirectionalLights_colors_green");
        spinbox_renderer_lighting_DirectionalLights_colors_green->setReadOnly(true);
        spinbox_renderer_lighting_DirectionalLights_colors_green->setMaximum(255);

        gridLayout_21->addWidget(spinbox_renderer_lighting_DirectionalLights_colors_green, 6, 1, 1, 1);

        hslider_renderer_lighting_DirectionalLights_colors_red = new QSlider(tab_13);
        hslider_renderer_lighting_DirectionalLights_colors_red->setObjectName("hslider_renderer_lighting_DirectionalLights_colors_red");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_DirectionalLights_colors_red->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_DirectionalLights_colors_red->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_DirectionalLights_colors_red->setMaximum(255);
        hslider_renderer_lighting_DirectionalLights_colors_red->setValue(0);
        hslider_renderer_lighting_DirectionalLights_colors_red->setOrientation(Qt::Horizontal);

        gridLayout_21->addWidget(hslider_renderer_lighting_DirectionalLights_colors_red, 4, 0, 1, 1);

        label_15 = new QLabel(tab_13);
        label_15->setObjectName("label_15");

        gridLayout_21->addWidget(label_15, 7, 0, 1, 1);


        verticalLayout_26->addLayout(gridLayout_21);

        label_17 = new QLabel(tab_13);
        label_17->setObjectName("label_17");

        verticalLayout_26->addWidget(label_17);

        dspinbox_renderer_lighting_DirectionalLights_intensity = new QDoubleSpinBox(tab_13);
        dspinbox_renderer_lighting_DirectionalLights_intensity->setObjectName("dspinbox_renderer_lighting_DirectionalLights_intensity");

        verticalLayout_26->addWidget(dspinbox_renderer_lighting_DirectionalLights_intensity);

        tabWidget_6->addTab(tab_13, QString());

        verticalLayout_24->addWidget(tabWidget_6);

        button_renderer_lighting_DirectionalLights_add = new QPushButton(groupBox_10);
        button_renderer_lighting_DirectionalLights_add->setObjectName("button_renderer_lighting_DirectionalLights_add");

        verticalLayout_24->addWidget(button_renderer_lighting_DirectionalLights_add);

        button_renderer_lighting_DirectionalLights_delete = new QPushButton(groupBox_10);
        button_renderer_lighting_DirectionalLights_delete->setObjectName("button_renderer_lighting_DirectionalLights_delete");

        verticalLayout_24->addWidget(button_renderer_lighting_DirectionalLights_delete);


        verticalLayout_27->addWidget(groupBox_10);


        verticalLayout_28->addLayout(verticalLayout_27);

        tab_renderer_lighting->addTab(tab_12, QString());
        tab_11 = new QWidget();
        tab_11->setObjectName("tab_11");
        gridLayout_28 = new QGridLayout(tab_11);
        gridLayout_28->setObjectName("gridLayout_28");
        gridLayout_27 = new QGridLayout();
        gridLayout_27->setObjectName("gridLayout_27");
        groupBox_11 = new QGroupBox(tab_11);
        groupBox_11->setObjectName("groupBox_11");
        sizePolicy10.setHeightForWidth(groupBox_11->sizePolicy().hasHeightForWidth());
        groupBox_11->setSizePolicy(sizePolicy10);
        groupBox_11->setFocusPolicy(Qt::NoFocus);
        groupBox_11->setAcceptDrops(false);
        groupBox_11->setAutoFillBackground(false);
        groupBox_11->setFlat(false);
        groupBox_11->setCheckable(false);
        groupBox_11->setChecked(false);
        verticalLayout_29 = new QVBoxLayout(groupBox_11);
        verticalLayout_29->setObjectName("verticalLayout_29");
        tabWidget_7 = new QTabWidget(groupBox_11);
        tabWidget_7->setObjectName("tabWidget_7");
        sizePolicy6.setHeightForWidth(tabWidget_7->sizePolicy().hasHeightForWidth());
        tabWidget_7->setSizePolicy(sizePolicy6);
        tabWidget_7->setDocumentMode(false);
        tabWidget_7->setTabsClosable(false);
        tabWidget_7->setTabBarAutoHide(false);
        tab_15 = new QWidget();
        tab_15->setObjectName("tab_15");
        verticalLayout_30 = new QVBoxLayout(tab_15);
        verticalLayout_30->setObjectName("verticalLayout_30");
        gridLayout_24 = new QGridLayout();
        gridLayout_24->setObjectName("gridLayout_24");
        hslider_renderer_lighting_SpotLights_colors_red = new QSlider(tab_15);
        hslider_renderer_lighting_SpotLights_colors_red->setObjectName("hslider_renderer_lighting_SpotLights_colors_red");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_SpotLights_colors_red->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_SpotLights_colors_red->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_SpotLights_colors_red->setMaximum(255);
        hslider_renderer_lighting_SpotLights_colors_red->setValue(0);
        hslider_renderer_lighting_SpotLights_colors_red->setOrientation(Qt::Horizontal);

        gridLayout_24->addWidget(hslider_renderer_lighting_SpotLights_colors_red, 4, 0, 1, 1);

        label_18 = new QLabel(tab_15);
        label_18->setObjectName("label_18");

        gridLayout_24->addWidget(label_18, 3, 0, 1, 1);

        label_19 = new QLabel(tab_15);
        label_19->setObjectName("label_19");

        gridLayout_24->addWidget(label_19, 5, 0, 1, 1);

        spinbox_renderer_lighting_SpotLights_colors_green = new QSpinBox(tab_15);
        spinbox_renderer_lighting_SpotLights_colors_green->setObjectName("spinbox_renderer_lighting_SpotLights_colors_green");
        spinbox_renderer_lighting_SpotLights_colors_green->setReadOnly(true);
        spinbox_renderer_lighting_SpotLights_colors_green->setMaximum(255);

        gridLayout_24->addWidget(spinbox_renderer_lighting_SpotLights_colors_green, 6, 1, 1, 1);

        hslider_renderer_lighting_SpotLights_colors_blue = new QSlider(tab_15);
        hslider_renderer_lighting_SpotLights_colors_blue->setObjectName("hslider_renderer_lighting_SpotLights_colors_blue");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_SpotLights_colors_blue->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_SpotLights_colors_blue->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_SpotLights_colors_blue->setMaximum(255);
        hslider_renderer_lighting_SpotLights_colors_blue->setOrientation(Qt::Horizontal);

        gridLayout_24->addWidget(hslider_renderer_lighting_SpotLights_colors_blue, 8, 0, 1, 1);

        hslider_renderer_lighting_SpotLights_colors_green = new QSlider(tab_15);
        hslider_renderer_lighting_SpotLights_colors_green->setObjectName("hslider_renderer_lighting_SpotLights_colors_green");
        sizePolicy6.setHeightForWidth(hslider_renderer_lighting_SpotLights_colors_green->sizePolicy().hasHeightForWidth());
        hslider_renderer_lighting_SpotLights_colors_green->setSizePolicy(sizePolicy6);
        hslider_renderer_lighting_SpotLights_colors_green->setMaximum(255);
        hslider_renderer_lighting_SpotLights_colors_green->setOrientation(Qt::Horizontal);

        gridLayout_24->addWidget(hslider_renderer_lighting_SpotLights_colors_green, 6, 0, 1, 1);

        spinbox_renderer_lighting_SpotLights_colors_red = new QSpinBox(tab_15);
        spinbox_renderer_lighting_SpotLights_colors_red->setObjectName("spinbox_renderer_lighting_SpotLights_colors_red");
        spinbox_renderer_lighting_SpotLights_colors_red->setReadOnly(true);
        spinbox_renderer_lighting_SpotLights_colors_red->setMaximum(255);

        gridLayout_24->addWidget(spinbox_renderer_lighting_SpotLights_colors_red, 4, 1, 1, 1);

        spinbox_renderer_lighting_SpotLights_colors_blue = new QSpinBox(tab_15);
        spinbox_renderer_lighting_SpotLights_colors_blue->setObjectName("spinbox_renderer_lighting_SpotLights_colors_blue");
        spinbox_renderer_lighting_SpotLights_colors_blue->setReadOnly(true);
        spinbox_renderer_lighting_SpotLights_colors_blue->setMaximum(255);

        gridLayout_24->addWidget(spinbox_renderer_lighting_SpotLights_colors_blue, 8, 1, 1, 1);

        label_20 = new QLabel(tab_15);
        label_20->setObjectName("label_20");

        gridLayout_24->addWidget(label_20, 7, 0, 1, 1);


        verticalLayout_30->addLayout(gridLayout_24);

        label_21 = new QLabel(tab_15);
        label_21->setObjectName("label_21");

        verticalLayout_30->addWidget(label_21);

        dspinbox_renderer_lighting_SpotLights_intensity = new QDoubleSpinBox(tab_15);
        dspinbox_renderer_lighting_SpotLights_intensity->setObjectName("dspinbox_renderer_lighting_SpotLights_intensity");

        verticalLayout_30->addWidget(dspinbox_renderer_lighting_SpotLights_intensity);

        tabWidget_7->addTab(tab_15, QString());
        tab_16 = new QWidget();
        tab_16->setObjectName("tab_16");
        gridLayout_25 = new QGridLayout(tab_16);
        gridLayout_25->setObjectName("gridLayout_25");
        gridLayout_26 = new QGridLayout();
        gridLayout_26->setObjectName("gridLayout_26");
        label_23 = new QLabel(tab_16);
        label_23->setObjectName("label_23");

        gridLayout_26->addWidget(label_23, 0, 0, 1, 1);

        dspinbox_renderer_lighting_SpotLights_angle = new QDoubleSpinBox(tab_16);
        dspinbox_renderer_lighting_SpotLights_angle->setObjectName("dspinbox_renderer_lighting_SpotLights_angle");
        dspinbox_renderer_lighting_SpotLights_angle->setDecimals(9);
        dspinbox_renderer_lighting_SpotLights_angle->setMaximum(10000000000000.000000000000000);

        gridLayout_26->addWidget(dspinbox_renderer_lighting_SpotLights_angle, 1, 0, 1, 1);

        dial_renderer_lighting_SpotLights_angle = new QDial(tab_16);
        dial_renderer_lighting_SpotLights_angle->setObjectName("dial_renderer_lighting_SpotLights_angle");
        QSizePolicy sizePolicy11(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy11.setHorizontalStretch(0);
        sizePolicy11.setVerticalStretch(0);
        sizePolicy11.setHeightForWidth(dial_renderer_lighting_SpotLights_angle->sizePolicy().hasHeightForWidth());
        dial_renderer_lighting_SpotLights_angle->setSizePolicy(sizePolicy11);
        dial_renderer_lighting_SpotLights_angle->setMaximum(180);
        dial_renderer_lighting_SpotLights_angle->setOrientation(Qt::Horizontal);
        dial_renderer_lighting_SpotLights_angle->setInvertedAppearance(false);
        dial_renderer_lighting_SpotLights_angle->setInvertedControls(false);

        gridLayout_26->addWidget(dial_renderer_lighting_SpotLights_angle, 2, 0, 1, 1);


        gridLayout_25->addLayout(gridLayout_26, 0, 0, 1, 1);

        tabWidget_7->addTab(tab_16, QString());

        verticalLayout_29->addWidget(tabWidget_7);

        button_renderer_lighting_SpotLights_add = new QPushButton(groupBox_11);
        button_renderer_lighting_SpotLights_add->setObjectName("button_renderer_lighting_SpotLights_add");

        verticalLayout_29->addWidget(button_renderer_lighting_SpotLights_add);

        button_renderer_lighting_SpotLights_delete = new QPushButton(groupBox_11);
        button_renderer_lighting_SpotLights_delete->setObjectName("button_renderer_lighting_SpotLights_delete");

        verticalLayout_29->addWidget(button_renderer_lighting_SpotLights_delete);


        gridLayout_27->addWidget(groupBox_11, 0, 0, 1, 1);


        gridLayout_28->addLayout(gridLayout_27, 0, 0, 1, 1);

        tab_renderer_lighting->addTab(tab_11, QString());
        tab_10 = new QWidget();
        tab_10->setObjectName("tab_10");
        tab_renderer_lighting->addTab(tab_10, QString());

        gridLayout_16->addWidget(tab_renderer_lighting, 3, 2, 1, 1);


        gridLayout_18->addLayout(gridLayout_16, 1, 0, 1, 1);

        tabWidget->addTab(tab_2, QString());
        tab_6 = new QWidget();
        tab_6->setObjectName("tab_6");
        horizontalLayout_6 = new QHBoxLayout(tab_6);
        horizontalLayout_6->setObjectName("horizontalLayout_6");
        verticalLayout_12 = new QVBoxLayout();
        verticalLayout_12->setObjectName("verticalLayout_12");
        groupBox_7 = new QGroupBox(tab_6);
        groupBox_7->setObjectName("groupBox_7");

        verticalLayout_12->addWidget(groupBox_7);

        groupBox_6 = new QGroupBox(tab_6);
        groupBox_6->setObjectName("groupBox_6");

        verticalLayout_12->addWidget(groupBox_6);


        horizontalLayout_6->addLayout(verticalLayout_12);

        tabWidget->addTab(tab_6, QString());

        gridLayout_6->addWidget(tabWidget, 2, 0, 1, 1);

        gridLayout_32 = new QGridLayout();
        gridLayout_32->setObjectName("gridLayout_32");
        renderer_view = new GLViewer(gl_renderer);
        renderer_view->setObjectName("renderer_view");
        renderer_view->setEnabled(true);
        sizePolicy.setHeightForWidth(renderer_view->sizePolicy().hasHeightForWidth());
        renderer_view->setSizePolicy(sizePolicy);
        renderer_view->setMouseTracking(true);
        renderer_view->setAcceptDrops(true);

        gridLayout_32->addWidget(renderer_view, 0, 0, 1, 1);

        verticalLayout_31 = new QVBoxLayout();
        verticalLayout_31->setObjectName("verticalLayout_31");
        renderer_scene_list = new SceneListView(gl_renderer);
        renderer_scene_list->setObjectName("renderer_scene_list");
        sizePolicy5.setHeightForWidth(renderer_scene_list->sizePolicy().hasHeightForWidth());
        renderer_scene_list->setSizePolicy(sizePolicy5);
        renderer_scene_list->setAnimated(true);
        renderer_scene_list->setHeaderHidden(false);
        renderer_scene_list->setColumnCount(1);
        renderer_scene_list->header()->setCascadingSectionResizes(false);

        verticalLayout_31->addWidget(renderer_scene_list);

        renderer_envmap_list = new EnvmapListDisplay(gl_renderer);
        renderer_envmap_list->setObjectName("renderer_envmap_list");
        sizePolicy5.setHeightForWidth(renderer_envmap_list->sizePolicy().hasHeightForWidth());
        renderer_envmap_list->setSizePolicy(sizePolicy5);
        renderer_envmap_list->setAcceptDrops(true);
        renderer_envmap_list->setFrameShape(QFrame::StyledPanel);
        renderer_envmap_list->setFrameShadow(QFrame::Sunken);

        verticalLayout_31->addWidget(renderer_envmap_list);


        gridLayout_32->addLayout(verticalLayout_31, 0, 1, 1, 1);


        gridLayout_6->addLayout(gridLayout_32, 2, 4, 1, 1);

        workspace->addTab(gl_renderer, QString());
        uv_editor = new QWidget();
        uv_editor->setObjectName("uv_editor");
        sizePolicy.setHeightForWidth(uv_editor->sizePolicy().hasHeightForWidth());
        uv_editor->setSizePolicy(sizePolicy);
        uv_editor->setLayoutDirection(Qt::LeftToRight);
        uv_editor->setAutoFillBackground(false);
        gridLayout_17 = new QGridLayout(uv_editor);
        gridLayout_17->setObjectName("gridLayout_17");
        gridLayout_14 = new QGridLayout();
        gridLayout_14->setObjectName("gridLayout_14");
        gridLayout_14->setSizeConstraint(QLayout::SetDefaultConstraint);
        meshes_list = new MeshListView(uv_editor);
        meshes_list->setObjectName("meshes_list");
        meshes_list->setFrameShape(QFrame::NoFrame);

        gridLayout_14->addWidget(meshes_list, 1, 1, 1, 1);

        gridLayout_8 = new QGridLayout();
        gridLayout_8->setObjectName("gridLayout_8");
        next_mesh_button = new QPushButton(uv_editor);
        next_mesh_button->setObjectName("next_mesh_button");
        sizePolicy7.setHeightForWidth(next_mesh_button->sizePolicy().hasHeightForWidth());
        next_mesh_button->setSizePolicy(sizePolicy7);

        gridLayout_8->addWidget(next_mesh_button, 0, 1, 1, 1);

        previous_mesh_button = new QPushButton(uv_editor);
        previous_mesh_button->setObjectName("previous_mesh_button");
        sizePolicy7.setHeightForWidth(previous_mesh_button->sizePolicy().hasHeightForWidth());
        previous_mesh_button->setSizePolicy(sizePolicy7);

        gridLayout_8->addWidget(previous_mesh_button, 0, 0, 1, 1);


        gridLayout_14->addLayout(gridLayout_8, 0, 1, 1, 1);


        gridLayout_17->addLayout(gridLayout_14, 0, 2, 1, 1);

        gridLayout_15 = new QGridLayout();
        gridLayout_15->setObjectName("gridLayout_15");
        gridLayout_15->setSizeConstraint(QLayout::SetDefaultConstraint);
        uv_projection = new QGraphicsView(uv_editor);
        uv_projection->setObjectName("uv_projection");
        sizePolicy.setHeightForWidth(uv_projection->sizePolicy().hasHeightForWidth());
        uv_projection->setSizePolicy(sizePolicy);
        uv_projection->setFrameShape(QFrame::NoFrame);
        uv_projection->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContentsOnFirstShow);
        uv_projection->setCacheMode(QGraphicsView::CacheNone);
        uv_projection->setResizeAnchor(QGraphicsView::AnchorViewCenter);

        gridLayout_15->addWidget(uv_projection, 0, 1, 1, 1);


        gridLayout_17->addLayout(gridLayout_15, 0, 0, 1, 1);

        workspace->addTab(uv_editor, QString());
        rt_renderer = new QWidget();
        rt_renderer->setObjectName("rt_renderer");
        gridLayout_30 = new QGridLayout(rt_renderer);
        gridLayout_30->setObjectName("gridLayout_30");
        gridLayout_29 = new QGridLayout();
        gridLayout_29->setObjectName("gridLayout_29");

        gridLayout_30->addLayout(gridLayout_29, 0, 0, 1, 1);

        workspace->addTab(rt_renderer, QString());

        gridLayout_4->addWidget(workspace, 0, 0, 1, 1);


        gridLayout->addLayout(gridLayout_4, 1, 0, 1, 1);


        verticalLayout->addLayout(gridLayout);


        gridLayout_2->addLayout(verticalLayout, 1, 0, 1, 1);

        use_gpu = new QCheckBox(centralwidget);
        use_gpu->setObjectName("use_gpu");
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/new/prefix1/nvidia-cuda21.png"), QSize(), QIcon::Normal, QIcon::Off);
        use_gpu->setIcon(icon2);
        use_gpu->setIconSize(QSize(32, 16));

        gridLayout_2->addWidget(use_gpu, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 1528, 20));
        menuFiles = new QMenu(menubar);
        menuFiles->setObjectName("menuFiles");
        menuEdit = new QMenu(menubar);
        menuEdit->setObjectName("menuEdit");
        menuTools = new QMenu(menubar);
        menuTools->setObjectName("menuTools");
        menuHelp = new QMenu(menubar);
        menuHelp->setObjectName("menuHelp");
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuFiles->menuAction());
        menubar->addAction(menuEdit->menuAction());
        menubar->addAction(menuTools->menuAction());
        menubar->addAction(menuHelp->menuAction());
        menuFiles->addAction(actionNew_Project);
        menuFiles->addAction(actionSave_project);
        menuFiles->addAction(actionImport_image);
        menuFiles->addAction(actionSave_image);
        menuFiles->addAction(actionImport_3D_model);
        menuFiles->addAction(actionImport_Environment_Map);
        menuFiles->addAction(actionExit);
        menuEdit->addAction(actionUndo);
        menuEdit->addAction(actionRedo);
        menuHelp->addAction(actionDocumentation);
        menuHelp->addAction(actionAxomae_version);

        retranslateUi(MainWindow);
        QObject::connect(actionExit, &QAction::triggered, MainWindow, qOverload<>(&QMainWindow::close));

        workspace->setCurrentIndex(2);
        renderMaterials->setCurrentIndex(0);
        tabWidget_2->setCurrentIndex(0);
        tabWidget_3->setCurrentIndex(0);
        tabWidget->setCurrentIndex(0);
        tab_renderer_lighting->setCurrentIndex(0);
        tabWidget_4->setCurrentIndex(0);
        tabWidget_6->setCurrentIndex(0);
        tabWidget_7->setCurrentIndex(0);
        previous_mesh_button->setDefault(false);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Axomae", nullptr));
        actionNew_Project->setText(QCoreApplication::translate("MainWindow", "&New Project", nullptr));
        actionImport_image->setText(QCoreApplication::translate("MainWindow", "&Import image", nullptr));
        actionSave_image->setText(QCoreApplication::translate("MainWindow", "&Save image", nullptr));
        actionSave_project->setText(QCoreApplication::translate("MainWindow", "Save &project", nullptr));
        actionExit->setText(QCoreApplication::translate("MainWindow", "&Exit", nullptr));
        actionDocumentation->setText(QCoreApplication::translate("MainWindow", "&Documentation", nullptr));
        actionAxomae_version->setText(QCoreApplication::translate("MainWindow", "&Axomae version", nullptr));
        actionUndo->setText(QCoreApplication::translate("MainWindow", "&Undo                            ", nullptr));
        actionRedo->setText(QCoreApplication::translate("MainWindow", "&Redo", nullptr));
        actionImport_3D_model->setText(QCoreApplication::translate("MainWindow", "Import &3D Model", nullptr));
        actionImport_3D_model->setIconText(QCoreApplication::translate("MainWindow", "Import 3D model", nullptr));
        actionImport_Environment_Map->setText(QCoreApplication::translate("MainWindow", "Import Environment Map", nullptr));
        undo_button->setText(QCoreApplication::translate("MainWindow", "undo", nullptr));
        redo_button->setText(QCoreApplication::translate("MainWindow", "redo", nullptr));
        greyscale_opt->setTitle(QCoreApplication::translate("MainWindow", "Greyscale options", nullptr));
        use_average->setText(QCoreApplication::translate("MainWindow", "Use a&verage", nullptr));
        use_luminance->setText(QCoreApplication::translate("MainWindow", "Use luminance", nullptr));
        height_opt->setTitle(QCoreApplication::translate("MainWindow", "Height options", nullptr));
        use_scharr->setText(QCoreApplication::translate("MainWindow", "Use Scharr", nullptr));
        use_sobel->setText(QCoreApplication::translate("MainWindow", "Use Sobel", nullptr));
        use_prewitt->setText(QCoreApplication::translate("MainWindow", "Use Prewitt", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab), QCoreApplication::translate("MainWindow", "Kernels", nullptr));
        sharpen_radio->setText(QCoreApplication::translate("MainWindow", "Sharpen", nullptr));
        sharpen_masking_radio->setText(QCoreApplication::translate("MainWindow", "Unsharp masking", nullptr));
        sharpen_button->setText(QCoreApplication::translate("MainWindow", "Sharpen", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(verticalLayout_16), QCoreApplication::translate("MainWindow", "Sharpen", nullptr));
        box_blur_radio->setText(QCoreApplication::translate("MainWindow", "Box Blur", nullptr));
        gaussian_5_5_radio->setText(QCoreApplication::translate("MainWindow", "Gaussian Blur (5x5)", nullptr));
        gaussian_3_3_radio->setText(QCoreApplication::translate("MainWindow", "Gaussian Blur (3x3)", nullptr));
#if QT_CONFIG(tooltip)
        smooth_dial->setToolTip(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Will compute successive iterations of the Blur function</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
        smooth_button->setText(QCoreApplication::translate("MainWindow", "Smooth", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(smoother_grid), QCoreApplication::translate("MainWindow", "Smooth", nullptr));
        normal_opt->setTitle(QCoreApplication::translate("MainWindow", "Normals options", nullptr));
        use_objectSpace->setText(QCoreApplication::translate("MainWindow", "Ob&ject space", nullptr));
        use_tangentSpace->setText(QCoreApplication::translate("MainWindow", "Tangent space", nullptr));
        nmap_factor_opt->setTitle(QCoreApplication::translate("MainWindow", "Factor", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Attenuation", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "Factor", nullptr));
        dudv_opt->setTitle(QCoreApplication::translate("MainWindow", "Distortion options", nullptr));
        compute_dudv->setText(QCoreApplication::translate("MainWindow", "Compute distortion map", nullptr));
        nmap_factor_opt_2->setTitle(QCoreApplication::translate("MainWindow", "Factor", nullptr));
        renderMaterials->setTabText(renderMaterials->indexOf(functions), QCoreApplication::translate("MainWindow", "Texture Tools", nullptr));
        workspace->setTabText(workspace->indexOf(texture), QCoreApplication::translate("MainWindow", "Normal Mapping", nullptr));
#if QT_CONFIG(whatsthis)
        workspace->setTabWhatsThis(workspace->indexOf(texture), QCoreApplication::translate("MainWindow", "Display current loaded image", nullptr));
#endif // QT_CONFIG(whatsthis)
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Irradiance Dimensions", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "Texture height", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Texture width", nullptr));
        bake_texture->setText(QCoreApplication::translate("MainWindow", "Bake", nullptr));
        tabWidget_3->setTabText(tabWidget_3->indexOf(tab_4), QCoreApplication::translate("MainWindow", "Irradiance map", nullptr));
        workspace->setTabText(workspace->indexOf(tab_3), QCoreApplication::translate("MainWindow", "Cube Mapping", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "Camera", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "Gamma", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Exposure", nullptr));
        reset_camera_button->setText(QCoreApplication::translate("MainWindow", "Reset Camera", nullptr));
        groupBox_4->setTitle(QCoreApplication::translate("MainWindow", "Post-Processing Effects", nullptr));
        set_standard_post_p->setText(QCoreApplication::translate("MainWindow", "No Post-Processing", nullptr));
        set_edge_post_p->setText(QCoreApplication::translate("MainWindow", "Edge", nullptr));
        set_sharpen_post_p->setText(QCoreApplication::translate("MainWindow", "Sharpen", nullptr));
        set_blurr_post_p->setText(QCoreApplication::translate("MainWindow", "Blurr", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(PostProTab), QCoreApplication::translate("MainWindow", "Screen", nullptr));
        groupBox_3->setTitle(QCoreApplication::translate("MainWindow", "Polygon Display", nullptr));
        rasterize_point_button->setText(QCoreApplication::translate("MainWindow", "Point", nullptr));
        rasterize_wireframe_button->setText(QCoreApplication::translate("MainWindow", "Wireframe", nullptr));
        rasterize_fill_button->setText(QCoreApplication::translate("MainWindow", "Fill", nullptr));
        groupBox_5->setTitle(QCoreApplication::translate("MainWindow", "Bounding Boxes", nullptr));
        rasterize_display_bbox_checkbox->setText(QCoreApplication::translate("MainWindow", "Display", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_5), QCoreApplication::translate("MainWindow", "Rasterization", nullptr));
        groupBox_9->setTitle(QCoreApplication::translate("MainWindow", "Configuration", nullptr));
        label_7->setText(QCoreApplication::translate("MainWindow", "Red", nullptr));
        label_8->setText(QCoreApplication::translate("MainWindow", "Green", nullptr));
        label_9->setText(QCoreApplication::translate("MainWindow", "Blue", nullptr));
        label_16->setText(QCoreApplication::translate("MainWindow", "Intensity", nullptr));
        tabWidget_4->setTabText(tabWidget_4->indexOf(tab_7), QCoreApplication::translate("MainWindow", "Colors", nullptr));
        label_11->setText(QCoreApplication::translate("MainWindow", "Linear", nullptr));
        label_10->setText(QCoreApplication::translate("MainWindow", "Constant", nullptr));
        label_12->setText(QCoreApplication::translate("MainWindow", "Quadratic", nullptr));
        tabWidget_4->setTabText(tabWidget_4->indexOf(tab_8), QCoreApplication::translate("MainWindow", "Attenuation", nullptr));
        button_renderer_lighting_PointLights_add->setText(QCoreApplication::translate("MainWindow", "Add", nullptr));
        button_renderer_lighting_PointLights_delete->setText(QCoreApplication::translate("MainWindow", "Delete", nullptr));
        tab_renderer_lighting->setTabText(tab_renderer_lighting->indexOf(tab_9), QCoreApplication::translate("MainWindow", "Point Lights", nullptr));
        groupBox_10->setTitle(QCoreApplication::translate("MainWindow", "Configuration", nullptr));
        label_13->setText(QCoreApplication::translate("MainWindow", "Red", nullptr));
        label_14->setText(QCoreApplication::translate("MainWindow", "Green", nullptr));
        label_15->setText(QCoreApplication::translate("MainWindow", "Blue", nullptr));
        label_17->setText(QCoreApplication::translate("MainWindow", "Intensity", nullptr));
        tabWidget_6->setTabText(tabWidget_6->indexOf(tab_13), QCoreApplication::translate("MainWindow", "Colors", nullptr));
        button_renderer_lighting_DirectionalLights_add->setText(QCoreApplication::translate("MainWindow", "Add", nullptr));
        button_renderer_lighting_DirectionalLights_delete->setText(QCoreApplication::translate("MainWindow", "Delete", nullptr));
        tab_renderer_lighting->setTabText(tab_renderer_lighting->indexOf(tab_12), QCoreApplication::translate("MainWindow", "Directional Lights", nullptr));
        groupBox_11->setTitle(QCoreApplication::translate("MainWindow", "Configuration", nullptr));
        label_18->setText(QCoreApplication::translate("MainWindow", "Red", nullptr));
        label_19->setText(QCoreApplication::translate("MainWindow", "Green", nullptr));
        label_20->setText(QCoreApplication::translate("MainWindow", "Blue", nullptr));
        label_21->setText(QCoreApplication::translate("MainWindow", "Intensity", nullptr));
        tabWidget_7->setTabText(tabWidget_7->indexOf(tab_15), QCoreApplication::translate("MainWindow", "Colors", nullptr));
        label_23->setText(QCoreApplication::translate("MainWindow", "Angle", nullptr));
        tabWidget_7->setTabText(tabWidget_7->indexOf(tab_16), QCoreApplication::translate("MainWindow", "Options", nullptr));
        button_renderer_lighting_SpotLights_add->setText(QCoreApplication::translate("MainWindow", "Add", nullptr));
        button_renderer_lighting_SpotLights_delete->setText(QCoreApplication::translate("MainWindow", "Delete", nullptr));
        tab_renderer_lighting->setTabText(tab_renderer_lighting->indexOf(tab_11), QCoreApplication::translate("MainWindow", "Spot Lights", nullptr));
        tab_renderer_lighting->setTabText(tab_renderer_lighting->indexOf(tab_10), QCoreApplication::translate("MainWindow", "Area Lights", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QCoreApplication::translate("MainWindow", "Lighting", nullptr));
        groupBox_7->setTitle(QCoreApplication::translate("MainWindow", "Object Transformations", nullptr));
        groupBox_6->setTitle(QCoreApplication::translate("MainWindow", "Geometry ", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_6), QCoreApplication::translate("MainWindow", "Scene", nullptr));
        QTreeWidgetItem *___qtreewidgetitem = renderer_scene_list->headerItem();
        ___qtreewidgetitem->setText(0, QCoreApplication::translate("MainWindow", "Scene", nullptr));
        workspace->setTabText(workspace->indexOf(gl_renderer), QCoreApplication::translate("MainWindow", "Renderer", nullptr));
#if QT_CONFIG(tooltip)
        meshes_list->setToolTip(QCoreApplication::translate("MainWindow", "<html><head/><body><p>List all meshes names in the 3D model</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
        next_mesh_button->setText(QCoreApplication::translate("MainWindow", "Next", nullptr));
        previous_mesh_button->setText(QCoreApplication::translate("MainWindow", "Previous", nullptr));
        workspace->setTabText(workspace->indexOf(uv_editor), QCoreApplication::translate("MainWindow", "UV editor", nullptr));
        workspace->setTabText(workspace->indexOf(rt_renderer), QCoreApplication::translate("MainWindow", "RT Renderer", nullptr));
        use_gpu->setText(QCoreApplication::translate("MainWindow", "GPGPU", nullptr));
        menuFiles->setTitle(QCoreApplication::translate("MainWindow", "Fi&les", nullptr));
        menuEdit->setTitle(QCoreApplication::translate("MainWindow", "Edit", nullptr));
        menuTools->setTitle(QCoreApplication::translate("MainWindow", "Tools", nullptr));
        menuHelp->setTitle(QCoreApplication::translate("MainWindow", "Help", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAIN_WINDOW_H
