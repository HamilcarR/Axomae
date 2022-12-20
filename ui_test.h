/********************************************************************************
** Form generated from reading UI file 'test.ui'
**
** Created by: Qt User Interface Compiler version 5.15.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TEST_H
#define UI_TEST_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

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
    QAction *actionCancel;
    QAction *actionRedo;
    QAction *actionImport_3D_model;
    QWidget *centralwidget;
    QGridLayout *gridLayout_2;
    QProgressBar *progressBar;
    QVBoxLayout *verticalLayout;
    QTabWidget *renderMaterials;
    QWidget *functions;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *use_gpu;
    QGroupBox *greyscale_opt;
    QVBoxLayout *verticalLayout_8;
    QRadioButton *use_average;
    QRadioButton *use_luminance;
    QGroupBox *height_opt;
    QVBoxLayout *verticalLayout_3;
    QRadioButton *use_scharr;
    QRadioButton *use_sobel;
    QRadioButton *use_prewitt;
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
    QWidget *tools;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_7;
    QGridLayout *gridLayout_3;
    QLabel *label_3;
    QSpinBox *uv_height;
    QLabel *label_4;
    QSpinBox *uv_width;
    QPushButton *bake_texture;
    QWidget *widget;
    QGridLayout *gridLayout;
    QGridLayout *gridLayout_4;
    QTabWidget *renderer_tab;
    QWidget *texture;
    QGridLayout *gridLayout_9;
    QGraphicsView *dudv_image;
    QGraphicsView *normal_image;
    QGraphicsView *diffuse_image;
    QTabWidget *tabWidget;
    QWidget *tab_greyscale;
    QGridLayout *gridLayout_5;
    QGraphicsView *greyscale_image;
    QWidget *tab_heightmap;
    QGridLayout *gridLayout_10;
    QGraphicsView *height_image;
    QWidget *gl_renderer;
    QGridLayout *gridLayout_6;
    QWidget *uv_editor;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout_11;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_12;
    QGraphicsView *uv_projection;
    QOpenGLWidget *uv_viewer;
    QMenuBar *menubar;
    QMenu *menuFiles;
    QMenu *menuEdit;
    QMenu *menuTools;
    QMenu *menuHelp;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1528, 900);
        MainWindow->setStyleSheet(QString::fromUtf8("QMainWindow,QWidget {\n"
"	background-color :rgb(42, 42, 42);\n"
"	\n"
"}\n"
"\n"
""));
        actionNew_Project = new QAction(MainWindow);
        actionNew_Project->setObjectName(QString::fromUtf8("actionNew_Project"));
        actionImport_image = new QAction(MainWindow);
        actionImport_image->setObjectName(QString::fromUtf8("actionImport_image"));
        actionSave_image = new QAction(MainWindow);
        actionSave_image->setObjectName(QString::fromUtf8("actionSave_image"));
        actionSave_project = new QAction(MainWindow);
        actionSave_project->setObjectName(QString::fromUtf8("actionSave_project"));
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionDocumentation = new QAction(MainWindow);
        actionDocumentation->setObjectName(QString::fromUtf8("actionDocumentation"));
        actionAxomae_version = new QAction(MainWindow);
        actionAxomae_version->setObjectName(QString::fromUtf8("actionAxomae_version"));
        actionCancel = new QAction(MainWindow);
        actionCancel->setObjectName(QString::fromUtf8("actionCancel"));
        actionRedo = new QAction(MainWindow);
        actionRedo->setObjectName(QString::fromUtf8("actionRedo"));
        actionImport_3D_model = new QAction(MainWindow);
        actionImport_3D_model->setObjectName(QString::fromUtf8("actionImport_3D_model"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        gridLayout_2 = new QGridLayout(centralwidget);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        progressBar = new QProgressBar(centralwidget);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setValue(24);

        gridLayout_2->addWidget(progressBar, 1, 0, 1, 2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setSizeConstraint(QLayout::SetFixedSize);
        verticalLayout->setContentsMargins(-1, -1, 0, -1);
        renderMaterials = new QTabWidget(centralwidget);
        renderMaterials->setObjectName(QString::fromUtf8("renderMaterials"));
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
        functions->setObjectName(QString::fromUtf8("functions"));
        verticalLayout_2 = new QVBoxLayout(functions);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        use_gpu = new QCheckBox(functions);
        use_gpu->setObjectName(QString::fromUtf8("use_gpu"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/new/prefix1/nvidia-cuda21.png"), QSize(), QIcon::Normal, QIcon::Off);
        use_gpu->setIcon(icon);
        use_gpu->setIconSize(QSize(32, 16));

        verticalLayout_2->addWidget(use_gpu);

        greyscale_opt = new QGroupBox(functions);
        greyscale_opt->setObjectName(QString::fromUtf8("greyscale_opt"));
        greyscale_opt->setEnabled(true);
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(greyscale_opt->sizePolicy().hasHeightForWidth());
        greyscale_opt->setSizePolicy(sizePolicy2);
        greyscale_opt->setFlat(false);
        greyscale_opt->setCheckable(false);
        verticalLayout_8 = new QVBoxLayout(greyscale_opt);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        use_average = new QRadioButton(greyscale_opt);
        use_average->setObjectName(QString::fromUtf8("use_average"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(use_average->sizePolicy().hasHeightForWidth());
        use_average->setSizePolicy(sizePolicy3);
        QFont font;
        font.setFamily(QString::fromUtf8("MS Shell Dlg 2"));
        use_average->setFont(font);
        use_average->setChecked(true);

        verticalLayout_8->addWidget(use_average);

        use_luminance = new QRadioButton(greyscale_opt);
        use_luminance->setObjectName(QString::fromUtf8("use_luminance"));
        sizePolicy3.setHeightForWidth(use_luminance->sizePolicy().hasHeightForWidth());
        use_luminance->setSizePolicy(sizePolicy3);

        verticalLayout_8->addWidget(use_luminance);


        verticalLayout_2->addWidget(greyscale_opt);

        height_opt = new QGroupBox(functions);
        height_opt->setObjectName(QString::fromUtf8("height_opt"));
        verticalLayout_3 = new QVBoxLayout(height_opt);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        use_scharr = new QRadioButton(height_opt);
        use_scharr->setObjectName(QString::fromUtf8("use_scharr"));

        verticalLayout_3->addWidget(use_scharr);

        use_sobel = new QRadioButton(height_opt);
        use_sobel->setObjectName(QString::fromUtf8("use_sobel"));
        use_sobel->setChecked(true);

        verticalLayout_3->addWidget(use_sobel);

        use_prewitt = new QRadioButton(height_opt);
        use_prewitt->setObjectName(QString::fromUtf8("use_prewitt"));

        verticalLayout_3->addWidget(use_prewitt);


        verticalLayout_2->addWidget(height_opt);

        normal_opt = new QGroupBox(functions);
        normal_opt->setObjectName(QString::fromUtf8("normal_opt"));
        verticalLayout_5 = new QVBoxLayout(normal_opt);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        use_objectSpace = new QRadioButton(normal_opt);
        use_objectSpace->setObjectName(QString::fromUtf8("use_objectSpace"));
        use_objectSpace->setChecked(true);

        verticalLayout_5->addWidget(use_objectSpace);

        use_tangentSpace = new QRadioButton(normal_opt);
        use_tangentSpace->setObjectName(QString::fromUtf8("use_tangentSpace"));

        verticalLayout_5->addWidget(use_tangentSpace);

        nmap_factor_opt = new QGroupBox(normal_opt);
        nmap_factor_opt->setObjectName(QString::fromUtf8("nmap_factor_opt"));
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(nmap_factor_opt->sizePolicy().hasHeightForWidth());
        nmap_factor_opt->setSizePolicy(sizePolicy4);
        verticalLayout_4 = new QVBoxLayout(nmap_factor_opt);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        factor_nmap = new QDoubleSpinBox(nmap_factor_opt);
        factor_nmap->setObjectName(QString::fromUtf8("factor_nmap"));

        verticalLayout_4->addWidget(factor_nmap);

        label = new QLabel(nmap_factor_opt);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout_4->addWidget(label);

        attenuation_slider_nmap = new QSlider(nmap_factor_opt);
        attenuation_slider_nmap->setObjectName(QString::fromUtf8("attenuation_slider_nmap"));
        attenuation_slider_nmap->setMinimum(1);
        attenuation_slider_nmap->setMaximum(10);
        attenuation_slider_nmap->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(attenuation_slider_nmap);

        label_2 = new QLabel(nmap_factor_opt);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_4->addWidget(label_2);

        factor_slider_nmap = new QSlider(nmap_factor_opt);
        factor_slider_nmap->setObjectName(QString::fromUtf8("factor_slider_nmap"));
        factor_slider_nmap->setStyleSheet(QString::fromUtf8("QSlider {\n"
"qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 255, 255, 255), stop:0.1 rgba(255, 255, 255, 255), stop:0.2 rgba(255, 176, 176, 167), stop:0.3 rgba(255, 151, 151, 92), stop:0.4 rgba(255, 125, 125, 51), stop:0.5 rgba(255, 76, 76, 205), stop:0.52 rgba(255, 76, 76, 205), stop:0.6 rgba(255, 180, 180, 84), stop:1 rgba(255, 255, 255, 0))\n"
"\n"
"}"));
        factor_slider_nmap->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(factor_slider_nmap);


        verticalLayout_5->addWidget(nmap_factor_opt);


        verticalLayout_2->addWidget(normal_opt);

        dudv_opt = new QGroupBox(functions);
        dudv_opt->setObjectName(QString::fromUtf8("dudv_opt"));
        verticalLayout_7 = new QVBoxLayout(dudv_opt);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        compute_dudv = new QPushButton(dudv_opt);
        compute_dudv->setObjectName(QString::fromUtf8("compute_dudv"));

        verticalLayout_7->addWidget(compute_dudv);

        nmap_factor_opt_2 = new QGroupBox(dudv_opt);
        nmap_factor_opt_2->setObjectName(QString::fromUtf8("nmap_factor_opt_2"));
        sizePolicy4.setHeightForWidth(nmap_factor_opt_2->sizePolicy().hasHeightForWidth());
        nmap_factor_opt_2->setSizePolicy(sizePolicy4);
        verticalLayout_6 = new QVBoxLayout(nmap_factor_opt_2);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        factor_dudv = new QDoubleSpinBox(nmap_factor_opt_2);
        factor_dudv->setObjectName(QString::fromUtf8("factor_dudv"));

        verticalLayout_6->addWidget(factor_dudv);

        factor_slider_dudv = new QSlider(nmap_factor_opt_2);
        factor_slider_dudv->setObjectName(QString::fromUtf8("factor_slider_dudv"));
        factor_slider_dudv->setStyleSheet(QString::fromUtf8("QSlider {\n"
"qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 255, 255, 255), stop:0.1 rgba(255, 255, 255, 255), stop:0.2 rgba(255, 176, 176, 167), stop:0.3 rgba(255, 151, 151, 92), stop:0.4 rgba(255, 125, 125, 51), stop:0.5 rgba(255, 76, 76, 205), stop:0.52 rgba(255, 76, 76, 205), stop:0.6 rgba(255, 180, 180, 84), stop:1 rgba(255, 255, 255, 0))\n"
"\n"
"}"));
        factor_slider_dudv->setOrientation(Qt::Horizontal);

        verticalLayout_6->addWidget(factor_slider_dudv);


        verticalLayout_7->addWidget(nmap_factor_opt_2);


        verticalLayout_2->addWidget(dudv_opt);

        renderMaterials->addTab(functions, QString());
        tools = new QWidget();
        tools->setObjectName(QString::fromUtf8("tools"));
        groupBox = new QGroupBox(tools);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(0, 20, 147, 138));
        gridLayout_7 = new QGridLayout(groupBox);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_3->addWidget(label_3, 1, 0, 1, 1);

        uv_height = new QSpinBox(groupBox);
        uv_height->setObjectName(QString::fromUtf8("uv_height"));
        uv_height->setMinimum(50);
        uv_height->setMaximum(8096);

        gridLayout_3->addWidget(uv_height, 1, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_3->addWidget(label_4, 0, 0, 1, 1);

        uv_width = new QSpinBox(groupBox);
        uv_width->setObjectName(QString::fromUtf8("uv_width"));
        uv_width->setMinimum(50);
        uv_width->setMaximum(8096);

        gridLayout_3->addWidget(uv_width, 0, 1, 1, 1);


        gridLayout_7->addLayout(gridLayout_3, 0, 0, 1, 1);

        bake_texture = new QPushButton(groupBox);
        bake_texture->setObjectName(QString::fromUtf8("bake_texture"));

        gridLayout_7->addWidget(bake_texture, 1, 0, 1, 1);

        renderMaterials->addTab(tools, QString());
        widget = new QWidget();
        widget->setObjectName(QString::fromUtf8("widget"));
        renderMaterials->addTab(widget, QString());

        verticalLayout->addWidget(renderMaterials);


        gridLayout_2->addLayout(verticalLayout, 0, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        renderer_tab = new QTabWidget(centralwidget);
        renderer_tab->setObjectName(QString::fromUtf8("renderer_tab"));
        sizePolicy.setHeightForWidth(renderer_tab->sizePolicy().hasHeightForWidth());
        renderer_tab->setSizePolicy(sizePolicy);
        renderer_tab->setTabShape(QTabWidget::Triangular);
        renderer_tab->setMovable(true);
        renderer_tab->setTabBarAutoHide(true);
        texture = new QWidget();
        texture->setObjectName(QString::fromUtf8("texture"));
        gridLayout_9 = new QGridLayout(texture);
        gridLayout_9->setObjectName(QString::fromUtf8("gridLayout_9"));
        dudv_image = new QGraphicsView(texture);
        dudv_image->setObjectName(QString::fromUtf8("dudv_image"));
        dudv_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        dudv_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_9->addWidget(dudv_image, 2, 2, 1, 1);

        normal_image = new QGraphicsView(texture);
        normal_image->setObjectName(QString::fromUtf8("normal_image"));
        normal_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        normal_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_9->addWidget(normal_image, 2, 0, 1, 1);

        diffuse_image = new QGraphicsView(texture);
        diffuse_image->setObjectName(QString::fromUtf8("diffuse_image"));
        diffuse_image->setFrameShadow(QFrame::Plain);
        diffuse_image->setLineWidth(1);
        diffuse_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        diffuse_image->setDragMode(QGraphicsView::ScrollHandDrag);
        diffuse_image->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

        gridLayout_9->addWidget(diffuse_image, 0, 0, 1, 1);

        tabWidget = new QTabWidget(texture);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setTabPosition(QTabWidget::East);
        tabWidget->setTabShape(QTabWidget::Triangular);
        tabWidget->setElideMode(Qt::ElideLeft);
        tabWidget->setTabsClosable(false);
        tabWidget->setMovable(true);
        tabWidget->setTabBarAutoHide(true);
        tab_greyscale = new QWidget();
        tab_greyscale->setObjectName(QString::fromUtf8("tab_greyscale"));
        gridLayout_5 = new QGridLayout(tab_greyscale);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        greyscale_image = new QGraphicsView(tab_greyscale);
        greyscale_image->setObjectName(QString::fromUtf8("greyscale_image"));
        greyscale_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_5->addWidget(greyscale_image, 0, 0, 1, 1);

        tabWidget->addTab(tab_greyscale, QString());
        tab_heightmap = new QWidget();
        tab_heightmap->setObjectName(QString::fromUtf8("tab_heightmap"));
        gridLayout_10 = new QGridLayout(tab_heightmap);
        gridLayout_10->setObjectName(QString::fromUtf8("gridLayout_10"));
        height_image = new QGraphicsView(tab_heightmap);
        height_image->setObjectName(QString::fromUtf8("height_image"));
        height_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);

        gridLayout_10->addWidget(height_image, 0, 0, 1, 1);

        tabWidget->addTab(tab_heightmap, QString());

        gridLayout_9->addWidget(tabWidget, 0, 2, 1, 1);

        renderer_tab->addTab(texture, QString());
        gl_renderer = new QWidget();
        gl_renderer->setObjectName(QString::fromUtf8("gl_renderer"));
        sizePolicy2.setHeightForWidth(gl_renderer->sizePolicy().hasHeightForWidth());
        gl_renderer->setSizePolicy(sizePolicy2);
        gridLayout_6 = new QGridLayout(gl_renderer);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        renderer_tab->addTab(gl_renderer, QString());
        uv_editor = new QWidget();
        uv_editor->setObjectName(QString::fromUtf8("uv_editor"));
        sizePolicy.setHeightForWidth(uv_editor->sizePolicy().hasHeightForWidth());
        uv_editor->setSizePolicy(sizePolicy);
        uv_editor->setLayoutDirection(Qt::LeftToRight);
        uv_editor->setAutoFillBackground(false);
        verticalLayoutWidget = new QWidget(uv_editor);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(0, -1, 1061, 781));
        verticalLayout_11 = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        verticalLayout_11->setContentsMargins(0, 0, 0, 0);
        groupBox_2 = new QGroupBox(verticalLayoutWidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        sizePolicy.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy);
        groupBox_2->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        groupBox_2->setFlat(true);
        groupBox_2->setCheckable(false);
        horizontalLayout = new QHBoxLayout(groupBox_2);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        verticalLayout_12 = new QVBoxLayout();
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        verticalLayout_12->setSizeConstraint(QLayout::SetDefaultConstraint);
        uv_projection = new QGraphicsView(groupBox_2);
        uv_projection->setObjectName(QString::fromUtf8("uv_projection"));
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(uv_projection->sizePolicy().hasHeightForWidth());
        uv_projection->setSizePolicy(sizePolicy5);

        verticalLayout_12->addWidget(uv_projection);

        uv_viewer = new QOpenGLWidget(groupBox_2);
        uv_viewer->setObjectName(QString::fromUtf8("uv_viewer"));
        sizePolicy2.setHeightForWidth(uv_viewer->sizePolicy().hasHeightForWidth());
        uv_viewer->setSizePolicy(sizePolicy2);
        uv_viewer->setMaximumSize(QSize(16777215, 16777215));

        verticalLayout_12->addWidget(uv_viewer);


        horizontalLayout->addLayout(verticalLayout_12);


        verticalLayout_11->addWidget(groupBox_2);

        renderer_tab->addTab(uv_editor, QString());

        gridLayout_4->addWidget(renderer_tab, 0, 0, 1, 1);


        gridLayout->addLayout(gridLayout_4, 1, 0, 1, 1);


        gridLayout_2->addLayout(gridLayout, 0, 1, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1528, 26));
        menuFiles = new QMenu(menubar);
        menuFiles->setObjectName(QString::fromUtf8("menuFiles"));
        menuEdit = new QMenu(menubar);
        menuEdit->setObjectName(QString::fromUtf8("menuEdit"));
        menuTools = new QMenu(menubar);
        menuTools->setObjectName(QString::fromUtf8("menuTools"));
        menuHelp = new QMenu(menubar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuFiles->menuAction());
        menubar->addAction(menuEdit->menuAction());
        menubar->addAction(menuTools->menuAction());
        menubar->addAction(menuHelp->menuAction());
        menuFiles->addAction(actionNew_Project);
        menuFiles->addAction(actionImport_image);
        menuFiles->addAction(actionImport_3D_model);
        menuFiles->addAction(actionSave_image);
        menuFiles->addAction(actionSave_project);
        menuFiles->addAction(actionExit);
        menuEdit->addAction(actionCancel);
        menuEdit->addAction(actionRedo);
        menuHelp->addAction(actionDocumentation);
        menuHelp->addAction(actionAxomae_version);

        retranslateUi(MainWindow);
        QObject::connect(actionExit, SIGNAL(triggered(bool)), MainWindow, SLOT(close()));

        renderMaterials->setCurrentIndex(1);
        renderer_tab->setCurrentIndex(2);
        tabWidget->setCurrentIndex(1);


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
        actionCancel->setText(QCoreApplication::translate("MainWindow", "&Undo                            ", nullptr));
        actionRedo->setText(QCoreApplication::translate("MainWindow", "&Redo", nullptr));
        actionImport_3D_model->setText(QCoreApplication::translate("MainWindow", "Import &3D model", nullptr));
        use_gpu->setText(QCoreApplication::translate("MainWindow", "Use GPU", nullptr));
        greyscale_opt->setTitle(QCoreApplication::translate("MainWindow", "Greyscale options", nullptr));
        use_average->setText(QCoreApplication::translate("MainWindow", "Use a&verage", nullptr));
        use_luminance->setText(QCoreApplication::translate("MainWindow", "Use luminance", nullptr));
        height_opt->setTitle(QCoreApplication::translate("MainWindow", "Height options", nullptr));
        use_scharr->setText(QCoreApplication::translate("MainWindow", "Use Scharr", nullptr));
        use_sobel->setText(QCoreApplication::translate("MainWindow", "Use Sobel", nullptr));
        use_prewitt->setText(QCoreApplication::translate("MainWindow", "Use Prewitt", nullptr));
        normal_opt->setTitle(QCoreApplication::translate("MainWindow", "Normals options", nullptr));
        use_objectSpace->setText(QCoreApplication::translate("MainWindow", "Ob&ject space", nullptr));
        use_tangentSpace->setText(QCoreApplication::translate("MainWindow", "Tangent space", nullptr));
        nmap_factor_opt->setTitle(QCoreApplication::translate("MainWindow", "Factor", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Attenuation", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "Factor", nullptr));
        dudv_opt->setTitle(QCoreApplication::translate("MainWindow", "Distortion options", nullptr));
        compute_dudv->setText(QCoreApplication::translate("MainWindow", "Compute distortion map", nullptr));
        nmap_factor_opt_2->setTitle(QCoreApplication::translate("MainWindow", "Factor", nullptr));
        renderMaterials->setTabText(renderMaterials->indexOf(functions), QCoreApplication::translate("MainWindow", "functions", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Baking texture dimensions", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "Texture height", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Texture width", nullptr));
        bake_texture->setText(QCoreApplication::translate("MainWindow", "PushButton", nullptr));
        renderMaterials->setTabText(renderMaterials->indexOf(tools), QCoreApplication::translate("MainWindow", "tools", nullptr));
        renderMaterials->setTabText(renderMaterials->indexOf(widget), QCoreApplication::translate("MainWindow", "Materials", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_greyscale), QCoreApplication::translate("MainWindow", "greyscale", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_heightmap), QCoreApplication::translate("MainWindow", "Height map", nullptr));
        renderer_tab->setTabText(renderer_tab->indexOf(texture), QCoreApplication::translate("MainWindow", "textures", nullptr));
#if QT_CONFIG(whatsthis)
        renderer_tab->setTabWhatsThis(renderer_tab->indexOf(texture), QCoreApplication::translate("MainWindow", "Display current loaded image", nullptr));
#endif // QT_CONFIG(whatsthis)
        renderer_tab->setTabText(renderer_tab->indexOf(gl_renderer), QCoreApplication::translate("MainWindow", "renderer", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "GroupBox", nullptr));
        renderer_tab->setTabText(renderer_tab->indexOf(uv_editor), QCoreApplication::translate("MainWindow", "UV editor", nullptr));
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

#endif // UI_TEST_H
