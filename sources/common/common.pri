

HEADERS += $$PWD/constants.h \
            $$PWD/Factory.h \
            $$PWD/GenericException.h \
            $$PWD/UniformNames.h \
            $$PWD/Observer.h\
            $$PWD/axomae_utils.h\
            $$PWD/Visitor.h \
            $$PWD/IAxObject.h \



INCLUDEPATH += $$PWD/../debug \
                $$PWD/../thread \


include($$PWD/math/math.pri)