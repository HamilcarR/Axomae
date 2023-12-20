
HEADERS +=  $$PWD/GLViewer.h\
            $$PWD/MeshListView.h\
            $$PWD/SceneListView.h\
            $$PWD/SceneSelector.h\
            $$PWD/MaterialViewer.h\
            $$PWD/Window.h


SOURCES +=  $$PWD/GLViewer.cpp\
            $$PWD/MeshListView.cpp\
            $$PWD/SceneSelector.cpp\
            $$PWD/MaterialViewer.cpp\
            $$PWD/Window.cpp


FORMS += "$$PWD/Form/main_window.ui"

UI_DIR += "$$PWD/Form/"

INCLUDEPATH += $$PWD/