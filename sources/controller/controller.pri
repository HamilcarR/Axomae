SOURCES += $$PWD/GUIWindow.cpp \
            $$PWD/LightControllerUI.cpp \
            $$PWD/OP_ProgressStatus.cpp \
            $$PWD/TextureViewerController.cpp

HEADERS += $$PWD/GUIWindow.h \
            $$PWD/LightControllerUI.h\
            $$PWD/EnvmapController.h\
            $$PWD/Operator.h \
            $$PWD/OP_ProgressStatus.h \
            $$PWD/TextureViewerController.h

INCLUDEPATH += $$PWD/../config \
                $$PWD/../database \
                $$PWD/../database/model \
                $$PWD/../controller\
                $$PWD/../editor


include($$PWD/cmd/cmd.pri)