SOURCES += $$PWD/ImageManager.cpp \
            $$PWD/OfflineCubemapProcessing.cpp \

HEADERS += $$PWD/OfflineCubemapProcessing.h \
            $$PWD/ImageManager.h \
            $$PWD/image_utils.h 

INCLUDEPATH += $$PWD/../gpu/cuda \
                $$PWD/../gpu/opengl
                $$PWD/../common \

include($$PWD/cuda/cuda.pri)