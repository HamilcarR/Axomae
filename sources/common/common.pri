

HEADERS += $$PWD/constants.h \
            $$PWD/Factory.h \
            $$PWD/UniformNames.h \
            $$PWD/Observer.h\
            $$PWD/axomae_utils.h\
            $$PWD/Visitor.h \
            $$PWD/IAxObject.h \



INCLUDEPATH += $$PWD/../debug \
                $$PWD/../thread \


include($$PWD/math/math.pri)
include($$PWD/image/image.pri)
include($$PWD/exception/exception.pri)
