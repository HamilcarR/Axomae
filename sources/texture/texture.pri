SOURCES += $$PWD/Texture.cpp \
            $$PWD/TextureGroup.cpp \
            $$PWD/EnvmapTextureManager.cpp

HEADERS += $$PWD/Texture.h \
            $$PWD/TextureFactory.h \
            $$PWD/TextureGroup.h \
            $$PWD/GenericTextureProcessing.h\
            $$PWD/EnvmapTextureManager.h

INCLUDEPATH += $$PWD/../gpu/opengl/shader \
                $$PWD/../common \
                $$PWD/../database