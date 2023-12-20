SOURCES += $$PWD/Drawable.cpp \
            $$PWD/LightingSystem.cpp \ 
            $$PWD/Renderer.cpp \
            $$PWD/RenderPipeline.cpp 

HEADERS += $$PWD/Drawable.h \
            $$PWD/LightingSystem.h \
            $$PWD/Renderer.h \
            $$PWD/RendererEnums.h \
            $$PWD/RenderPipeline.h 

INCLUDEPATH += $$PWD/../camera \
                $$PWD/../gpu \
                $$PWD/../mesh \
                $$PWD/../texture