
HEADERS += $$PWD/ProgressStatusWidget.h\

SOURCES += $$PWD/ProgressStatusWidget.cpp\

FORMS += $$PWD/Form/main_window.ui \
         $$PWD/Form/texture_viewer.ui \

UI_DIR += "$$PWD/Form/"

include($$PWD/renderer/renderer.pri)
include($$PWD/UV/UV.pri)
include($$PWD/Form/Form.pri)
include($$PWD/image/image.pri)