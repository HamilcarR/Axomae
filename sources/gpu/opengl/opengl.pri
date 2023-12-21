SOURCES += $$PWD/CameraFrameBuffer.cpp \
            $$PWD/FrameBufferInterface.cpp \
            $$PWD/GLFrameBuffer.cpp \
            $$PWD/GLGeometryBuffer.cpp \
            $$PWD/GLRenderBuffer.cpp \
            $$PWD/RenderCubeMap.cpp \
            $$PWD/RenderQuad.cpp \

HEADERS += $$PWD/CameraFrameBuffer.h \
            $$PWD/FrameBufferInterface.h \
            $$PWD/GLFrameBuffer.h \
            $$PWD/GLGeometryBuffer.h \
            $$PWD/GLRenderBuffer.h \ 
            $$PWD/RenderCubeMap.h \
            $$PWD/RenderQuad.h \

INCLUDEPATH += $$PWD/../../scene \
                $$PWD/../../common \
                $$PWD/../../database \
                $$PWD/../../geometry \
                 

include($$PWD/shader/shader.pri)