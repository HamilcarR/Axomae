SOURCES += $$PWD/ImageImporter.cpp\
            $$PWD/Loader_scene.cpp\
            $$PWD/Loader_image.cpp\
            $$PWD/Loader_file.cpp\
            $$PWD/Logger.cpp

HEADERS += $$PWD/Loader.h\
            $$PWD/ImageImporter.h\
            $$PWD/LoaderSharedExceptions.h\
            $$PWD/Logger.h

INCLUDEPATH += $$PWD/../vendor \
                $$PWD/../scene \
                $$PWD/../thread \
                $$PWD/../gpu/opengl/shader \
                $$PWD/../texture \
                $$PWD/../common \
                $$PWD/../common/exception \
                $$PWD/../debug \
                
