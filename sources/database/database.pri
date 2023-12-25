SOURCES += $$PWD/INodeDatabase.cpp \
            $$PWD/LightingDatabase.cpp \
            $$PWD/ResourceDatabaseManager.cpp \  
            $$PWD/ShaderDatabase.cpp \
            $$PWD/TextureDatabase.cpp \
            $$PWD/ImageDatabase.cpp \

HEADERS += $$PWD/INodeDatabase.h \
            $$PWD/LightingDatabase.h \
            $$PWD/RenderingDatabaseInterface.h \
            $$PWD/ResourceDatabaseManager.h \
            $$PWD/ShaderDatabase.h \
            $$PWD/TextureDatabase.h \ 
            $$PWD/ImageDatabase.h\
            $$PWD/database_utils.h
            

INCLUDEPATH += $$PWD/../common \
                $$PWD/../database \
                $$PWD/../thread \
                $$PWD/../texture \
                $$PWD/../scene \
                $$PWD/../mesh \
                $$PWD/../editor/image\

include($$PWD/model/model.pri)