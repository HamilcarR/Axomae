
HEADERS += $$PWD/ProgressStatusWidget.h\
            $$PWD/TextureViewerWidget.h

SOURCES += $$PWD/ProgressStatusWidget.cpp\
            $$PWD/TextureViewerWidget.cpp

FORMS += $$PWD/Form/main_window.ui \
         $$PWD/Form/texture_viewer.ui \

UI_DIR += "$$PWD/Form/"

INCLUDEPATH += $$PWD/../common/image\


include($$PWD/renderer/renderer.pri)
include($$PWD/UV/UV.pri)
include($$PWD/Form/Form.pri)
include($$PWD/image/image.pri)