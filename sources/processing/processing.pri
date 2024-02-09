SOURCES += $$PWD/ImageManager.cpp \
            $$PWD/OfflineCubemapProcessing.cpp \
            $$PWD/ImageManager_UV.cpp

HEADERS += $$PWD/OfflineCubemapProcessing.h \
            $$PWD/ImageManager.h \

INCLUDEPATH += $$PWD/../gpu/cuda \
                $$PWD/../gpu/opengl\
                $$PWD/../common \
                $$PWD/../common/exception\

include($$PWD/cuda/cuda.pri)