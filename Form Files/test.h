/********************************************************************************
** Form generated from reading UI file 'test.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef TEST_H
#define TEST_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBox>
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
    QGridLayout *gridLayout_3;
    QToolBox *toolBox;
    QWidget *page;
    QWidget *page_2;
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
    QGridLayout *gridLayout_7;
    QMenuBar *menubar;
    QMenu *menuFiles;
    QMenu *menuEdit;
    QMenu *menuTools;
    QMenu *menuHelp;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1079, 900);
        MainWindow->setStyleSheet(QLatin1String("QMainWindow,QWidget {\n"
"	background-color :rgb(42, 42, 42);\n"
"	\n"
"}\n"
"\n"
""));
        actionNew_Project = new QAction(MainWindow);
        actionNew_Project->setObjectName(QStringLiteral("actionNew_Project"));
        actionImport_image = new QAction(MainWindow);
        actionImport_image->setObjectName(QStringLiteral("actionImport_image"));
        actionSave_image = new QAction(MainWindow);
        actionSave_image->setObjectName(QStringLiteral("actionSave_image"));
        actionSave_project = new QAction(MainWindow);
        actionSave_project->setObjectName(QStringLiteral("actionSave_project"));
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        actionDocumentation = new QAction(MainWindow);
        actionDocumentation->setObjectName(QStringLiteral("actionDocumentation"));
        actionAxomae_version = new QAction(MainWindow);
        actionAxomae_version->setObjectName(QStringLiteral("actionAxomae_version"));
        actionCancel = new QAction(MainWindow);
        actionCancel->setObjectName(QStringLiteral("actionCancel"));
        actionRedo = new QAction(MainWindow);
        actionRedo->setObjectName(QStringLiteral("actionRedo"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        gridLayout_2 = new QGridLayout(centralwidget);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        progressBar = new QProgressBar(centralwidget);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setValue(24);

        gridLayout_2->addWidget(progressBar, 1, 0, 1, 2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setSizeConstraint(QLayout::SetFixedSize);
        verticalLayout->setContentsMargins(-1, -1, 0, -1);
        renderMaterials = new QTabWidget(centralwidget);
        renderMaterials->setObjectName(QStringLiteral("renderMaterials"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(renderMaterials->sizePolicy().hasHeightForWidth());
        renderMaterials->setSizePolicy(sizePolicy1);
        renderMaterials->setStyleSheet(QStringLiteral(""));
        renderMaterials->setTabShape(QTabWidget::Triangular);
        renderMaterials->setMovable(true);
        renderMaterials->setTabBarAutoHide(true);
        functions = new QWidget();
        functions->setObjectName(QStringLiteral("functions"));
        verticalLayout_2 = new QVBoxLayout(functions);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        use_gpu = new QCheckBox(functions);
        use_gpu->setObjectName(QStringLiteral("use_gpu"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/new/prefix1/nvidia-cuda21.png"), QSize(), QIcon::Normal, QIcon::Off);
        use_gpu->setIcon(icon);
        use_gpu->setIconSize(QSize(32, 16));

        verticalLayout_2->addWidget(use_gpu);

        greyscale_opt = new QGroupBox(functions);
        greyscale_opt->setObjectName(QStringLiteral("greyscale_opt"));
        greyscale_opt->setEnabled(true);
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(greyscale_opt->sizePolicy().hasHeightForWidth());
        greyscale_opt->setSizePolicy(sizePolicy2);
        greyscale_opt->setFlat(false);
        greyscale_opt->setCheckable(false);
        verticalLayout_8 = new QVBoxLayout(greyscale_opt);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        use_average = new QRadioButton(greyscale_opt);
        use_average->setObjectName(QStringLiteral("use_average"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(use_average->sizePolicy().hasHeightForWidth());
        use_average->setSizePolicy(sizePolicy3);
        QFont font;
        font.setFamily(QStringLiteral("MS Shell Dlg 2"));
        use_average->setFont(font);
        use_average->setChecked(true);

        verticalLayout_8->addWidget(use_average);

        use_luminance = new QRadioButton(greyscale_opt);
        use_luminance->setObjectName(QStringLiteral("use_luminance"));
        sizePolicy3.setHeightForWidth(use_luminance->sizePolicy().hasHeightForWidth());
        use_luminance->setSizePolicy(sizePolicy3);

        verticalLayout_8->addWidget(use_luminance);


        verticalLayout_2->addWidget(greyscale_opt);

        height_opt = new QGroupBox(functions);
        height_opt->setObjectName(QStringLiteral("height_opt"));
        verticalLayout_3 = new QVBoxLayout(height_opt);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        use_scharr = new QRadioButton(height_opt);
        use_scharr->setObjectName(QStringLiteral("use_scharr"));

        verticalLayout_3->addWidget(use_scharr);

        use_sobel = new QRadioButton(height_opt);
        use_sobel->setObjectName(QStringLiteral("use_sobel"));
        use_sobel->setChecked(true);

        verticalLayout_3->addWidget(use_sobel);

        use_prewitt = new QRadioButton(height_opt);
        use_prewitt->setObjectName(QStringLiteral("use_prewitt"));

        verticalLayout_3->addWidget(use_prewitt);


        verticalLayout_2->addWidget(height_opt);

        normal_opt = new QGroupBox(functions);
        normal_opt->setObjectName(QStringLiteral("normal_opt"));
        verticalLayout_5 = new QVBoxLayout(normal_opt);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        use_objectSpace = new QRadioButton(normal_opt);
        use_objectSpace->setObjectName(QStringLiteral("use_objectSpace"));
        use_objectSpace->setChecked(true);

        verticalLayout_5->addWidget(use_objectSpace);

        use_tangentSpace = new QRadioButton(normal_opt);
        use_tangentSpace->setObjectName(QStringLiteral("use_tangentSpace"));

        verticalLayout_5->addWidget(use_tangentSpace);

        nmap_factor_opt = new QGroupBox(normal_opt);
        nmap_factor_opt->setObjectName(QStringLiteral("nmap_factor_opt"));
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(nmap_factor_opt->sizePolicy().hasHeightForWidth());
        nmap_factor_opt->setSizePolicy(sizePolicy4);
        verticalLayout_4 = new QVBoxLayout(nmap_factor_opt);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        factor_nmap = new QDoubleSpinBox(nmap_factor_opt);
        factor_nmap->setObjectName(QStringLiteral("factor_nmap"));

        verticalLayout_4->addWidget(factor_nmap);

        label = new QLabel(nmap_factor_opt);
        label->setObjectName(QStringLiteral("label"));

        verticalLayout_4->addWidget(label);

        attenuation_slider_nmap = new QSlider(nmap_factor_opt);
        attenuation_slider_nmap->setObjectName(QStringLiteral("attenuation_slider_nmap"));
        attenuation_slider_nmap->setMinimum(1);
        attenuation_slider_nmap->setMaximum(10);
        attenuation_slider_nmap->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(attenuation_slider_nmap);

        label_2 = new QLabel(nmap_factor_opt);
        label_2->setObjectName(QStringLiteral("label_2"));

        verticalLayout_4->addWidget(label_2);

        factor_slider_nmap = new QSlider(nmap_factor_opt);
        factor_slider_nmap->setObjectName(QStringLiteral("factor_slider_nmap"));
        factor_slider_nmap->setStyleSheet(QLatin1String("QSlider {\n"
"qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 255, 255, 255), stop:0.1 rgba(255, 255, 255, 255), stop:0.2 rgba(255, 176, 176, 167), stop:0.3 rgba(255, 151, 151, 92), stop:0.4 rgba(255, 125, 125, 51), stop:0.5 rgba(255, 76, 76, 205), stop:0.52 rgba(255, 76, 76, 205), stop:0.6 rgba(255, 180, 180, 84), stop:1 rgba(255, 255, 255, 0))\n"
"\n"
"}"));
        factor_slider_nmap->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(factor_slider_nmap);


        verticalLayout_5->addWidget(nmap_factor_opt);


        verticalLayout_2->addWidget(normal_opt);

        dudv_opt = new QGroupBox(functions);
        dudv_opt->setObjectName(QStringLiteral("dudv_opt"));
        verticalLayout_7 = new QVBoxLayout(dudv_opt);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        compute_dudv = new QPushButton(dudv_opt);
        compute_dudv->setObjectName(QStringLiteral("compute_dudv"));

        verticalLayout_7->addWidget(compute_dudv);

        nmap_factor_opt_2 = new QGroupBox(dudv_opt);
        nmap_factor_opt_2->setObjectName(QStringLiteral("nmap_factor_opt_2"));
        sizePolicy4.setHeightForWidth(nmap_factor_opt_2->sizePolicy().hasHeightForWidth());
        nmap_factor_opt_2->setSizePolicy(sizePolicy4);
        verticalLayout_6 = new QVBoxLayout(nmap_factor_opt_2);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        factor_dudv = new QDoubleSpinBox(nmap_factor_opt_2);
        factor_dudv->setObjectName(QStringLiteral("factor_dudv"));

        verticalLayout_6->addWidget(factor_dudv);

        factor_slider_dudv = new QSlider(nmap_factor_opt_2);
        factor_slider_dudv->setObjectName(QStringLiteral("factor_slider_dudv"));
        factor_slider_dudv->setStyleSheet(QLatin1String("QSlider {\n"
"qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 255, 255, 255), stop:0.1 rgba(255, 255, 255, 255), stop:0.2 rgba(255, 176, 176, 167), stop:0.3 rgba(255, 151, 151, 92), stop:0.4 rgba(255, 125, 125, 51), stop:0.5 rgba(255, 76, 76, 205), stop:0.52 rgba(255, 76, 76, 205), stop:0.6 rgba(255, 180, 180, 84), stop:1 rgba(255, 255, 255, 0))\n"
"\n"
"}"));
        factor_slider_dudv->setOrientation(Qt::Horizontal);

        verticalLayout_6->addWidget(factor_slider_dudv);


        verticalLayout_7->addWidget(nmap_factor_opt_2);


        verticalLayout_2->addWidget(dudv_opt);

        renderMaterials->addTab(functions, QString());
        tools = new QWidget();
        tools->setObjectName(QStringLiteral("tools"));
        gridLayout_3 = new QGridLayout(tools);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        toolBox = new QToolBox(tools);
        toolBox->setObjectName(QStringLiteral("toolBox"));
        page = new QWidget();
        page->setObjectName(QStringLiteral("page"));
        page->setGeometry(QRect(0, 0, 164, 720));
        toolBox->addItem(page, QStringLiteral("Page 1"));
        page_2 = new QWidget();
        page_2->setObjectName(QStringLiteral("page_2"));
        page_2->setGeometry(QRect(0, 0, 164, 720));
        toolBox->addItem(page_2, QStringLiteral("Page 2"));

        gridLayout_3->addWidget(toolBox, 0, 0, 1, 1);

        renderMaterials->addTab(tools, QString());
        widget = new QWidget();
        widget->setObjectName(QStringLiteral("widget"));
        renderMaterials->addTab(widget, QString());

        verticalLayout->addWidget(renderMaterials);


        gridLayout_2->addLayout(verticalLayout, 0, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        renderer_tab = new QTabWidget(centralwidget);
        renderer_tab->setObjectName(QStringLiteral("renderer_tab"));
        renderer_tab->setTabShape(QTabWidget::Triangular);
        renderer_tab->setMovable(true);
        renderer_tab->setTabBarAutoHide(true);
        texture = new QWidget();
        texture->setObjectName(QStringLiteral("texture"));
        gridLayout_9 = new QGridLayout(texture);
        gridLayout_9->setObjectName(QStringLiteral("gridLayout_9"));
        dudv_image = new QGraphicsView(texture);
        dudv_image->setObjectName(QStringLiteral("dudv_image"));
        dudv_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        dudv_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_9->addWidget(dudv_image, 2, 2, 1, 1);

        normal_image = new QGraphicsView(texture);
        normal_image->setObjectName(QStringLiteral("normal_image"));
        normal_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        normal_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_9->addWidget(normal_image, 2, 0, 1, 1);

        diffuse_image = new QGraphicsView(texture);
        diffuse_image->setObjectName(QStringLiteral("diffuse_image"));
        diffuse_image->setFrameShadow(QFrame::Plain);
        diffuse_image->setLineWidth(1);
        diffuse_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
        diffuse_image->setDragMode(QGraphicsView::ScrollHandDrag);
        diffuse_image->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

        gridLayout_9->addWidget(diffuse_image, 0, 0, 1, 1);

        tabWidget = new QTabWidget(texture);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setTabPosition(QTabWidget::East);
        tabWidget->setTabShape(QTabWidget::Triangular);
        tabWidget->setElideMode(Qt::ElideLeft);
        tabWidget->setTabsClosable(false);
        tabWidget->setMovable(true);
        tabWidget->setTabBarAutoHide(true);
        tab_greyscale = new QWidget();
        tab_greyscale->setObjectName(QStringLiteral("tab_greyscale"));
        gridLayout_5 = new QGridLayout(tab_greyscale);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        greyscale_image = new QGraphicsView(tab_greyscale);
        greyscale_image->setObjectName(QStringLiteral("greyscale_image"));
        greyscale_image->setDragMode(QGraphicsView::ScrollHandDrag);

        gridLayout_5->addWidget(greyscale_image, 0, 0, 1, 1);

        tabWidget->addTab(tab_greyscale, QString());
        tab_heightmap = new QWidget();
        tab_heightmap->setObjectName(QStringLiteral("tab_heightmap"));
        gridLayout_10 = new QGridLayout(tab_heightmap);
        gridLayout_10->setObjectName(QStringLiteral("gridLayout_10"));
        height_image = new QGraphicsView(tab_heightmap);
        height_image->setObjectName(QStringLiteral("height_image"));
        height_image->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);

        gridLayout_10->addWidget(height_image, 0, 0, 1, 1);

        tabWidget->addTab(tab_heightmap, QString());

        gridLayout_9->addWidget(tabWidget, 0, 2, 1, 1);

        renderer_tab->addTab(texture, QString());
        gl_renderer = new QWidget();
        gl_renderer->setObjectName(QStringLiteral("gl_renderer"));
        sizePolicy2.setHeightForWidth(gl_renderer->sizePolicy().hasHeightForWidth());
        gl_renderer->setSizePolicy(sizePolicy2);
        gridLayout_6 = new QGridLayout(gl_renderer);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        renderer_tab->addTab(gl_renderer, QString());
        uv_editor = new QWidget();
        uv_editor->setObjectName(QStringLiteral("uv_editor"));
        gridLayout_7 = new QGridLayout(uv_editor);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        renderer_tab->addTab(uv_editor, QString());

        gridLayout_4->addWidget(renderer_tab, 0, 0, 1, 1);


        gridLayout->addLayout(gridLayout_4, 1, 0, 1, 1);


        gridLayout_2->addLayout(gridLayout, 0, 1, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 1079, 21));
        menuFiles = new QMenu(menubar);
        menuFiles->setObjectName(QStringLiteral("menuFiles"));
        menuEdit = new QMenu(menubar);
        menuEdit->setObjectName(QStringLiteral("menuEdit"));
        menuTools = new QMenu(menubar);
        menuTools->setObjectName(QStringLiteral("menuTools"));
        menuHelp = new QMenu(menubar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuFiles->menuAction());
        menubar->addAction(menuEdit->menuAction());
        menubar->addAction(menuTools->menuAction());
        menubar->addAction(menuHelp->menuAction());
        menuFiles->addAction(actionNew_Project);
        menuFiles->addAction(actionImport_image);
        menuFiles->addAction(actionSave_image);
        menuFiles->addAction(actionSave_project);
        menuFiles->addAction(actionExit);
        menuEdit->addAction(actionCancel);
        menuEdit->addAction(actionRedo);
        menuHelp->addAction(actionDocumentation);
        menuHelp->addAction(actionAxomae_version);

        retranslateUi(MainWindow);
        QObject::connect(actionExit, SIGNAL(triggered(bool)), MainWindow, SLOT(close()));

        renderMaterials->setCurrentIndex(0);
        toolBox->setCurrentIndex(0);
        renderer_tab->setCurrentIndex(0);
        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Axomae", 0));
        actionNew_Project->setText(QApplication::translate("MainWindow", "New Project", 0));
        actionImport_image->setText(QApplication::translate("MainWindow", "Import image", 0));
        actionSave_image->setText(QApplication::translate("MainWindow", "Save image", 0));
        actionSave_project->setText(QApplication::translate("MainWindow", "Save project", 0));
        actionExit->setText(QApplication::translate("MainWindow", "Exit", 0));
        actionDocumentation->setText(QApplication::translate("MainWindow", "Documentation", 0));
        actionAxomae_version->setText(QApplication::translate("MainWindow", "Axomae version", 0));
        actionCancel->setText(QApplication::translate("MainWindow", "Undo                            ", 0));
        actionRedo->setText(QApplication::translate("MainWindow", "Redo", 0));
        use_gpu->setText(QApplication::translate("MainWindow", "Use GPU", 0));
        greyscale_opt->setTitle(QApplication::translate("MainWindow", "Greyscale options", 0));
        use_average->setText(QApplication::translate("MainWindow", "Use average", 0));
        use_luminance->setText(QApplication::translate("MainWindow", "Use luminance", 0));
        height_opt->setTitle(QApplication::translate("MainWindow", "Height options", 0));
        use_scharr->setText(QApplication::translate("MainWindow", "Use Scharr", 0));
        use_sobel->setText(QApplication::translate("MainWindow", "Use Sobel", 0));
        use_prewitt->setText(QApplication::translate("MainWindow", "Use Prewitt", 0));
        normal_opt->setTitle(QApplication::translate("MainWindow", "Normals options", 0));
        use_objectSpace->setText(QApplication::translate("MainWindow", "Object space", 0));
        use_tangentSpace->setText(QApplication::translate("MainWindow", "Tangent space", 0));
        nmap_factor_opt->setTitle(QApplication::translate("MainWindow", "Factor", 0));
        label->setText(QApplication::translate("MainWindow", "Attenuation", 0));
        label_2->setText(QApplication::translate("MainWindow", "Factor", 0));
        dudv_opt->setTitle(QApplication::translate("MainWindow", "Distortion options", 0));
        compute_dudv->setText(QApplication::translate("MainWindow", "Compute distortion map", 0));
        nmap_factor_opt_2->setTitle(QApplication::translate("MainWindow", "Factor", 0));
        renderMaterials->setTabText(renderMaterials->indexOf(functions), QApplication::translate("MainWindow", "functions", 0));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("MainWindow", "Page 1", 0));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("MainWindow", "Page 2", 0));
        renderMaterials->setTabText(renderMaterials->indexOf(tools), QApplication::translate("MainWindow", "tools", 0));
        renderMaterials->setTabText(renderMaterials->indexOf(widget), QApplication::translate("MainWindow", "Materials", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_greyscale), QApplication::translate("MainWindow", "greyscale", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_heightmap), QApplication::translate("MainWindow", "Height map", 0));
        renderer_tab->setTabText(renderer_tab->indexOf(texture), QApplication::translate("MainWindow", "textures", 0));
        renderer_tab->setTabWhatsThis(renderer_tab->indexOf(texture), QApplication::translate("MainWindow", "Display current loaded image", 0));
        renderer_tab->setTabText(renderer_tab->indexOf(gl_renderer), QApplication::translate("MainWindow", "renderer", 0));
        renderer_tab->setTabText(renderer_tab->indexOf(uv_editor), QApplication::translate("MainWindow", "UV editor", 0));
        menuFiles->setTitle(QApplication::translate("MainWindow", "Files", 0));
        menuEdit->setTitle(QApplication::translate("MainWindow", "Edit", 0));
        menuTools->setTitle(QApplication::translate("MainWindow", "Tools", 0));
        menuHelp->setTitle(QApplication::translate("MainWindow", "Help", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // TEST_H
