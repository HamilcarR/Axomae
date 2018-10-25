# HamilcarR #

TEMPLATE = app
TARGET = Axomae
QT += widgets
CONFIG += debug \
	opengl \
	warn_on \

	
#Configure these env variables before compiling#
CUDA_ARCH = sm_61
CUDA_SRC += kernels/*.cu


#Windows rules#
win32:{
QMAKE_CXXFLAGS += /std:c++17
}

#Linux rules#
linux:{
INCLUDEPATH +=	/usr/include/SDL2
QMAKE_DEFAULT_INCDIRS += -I/usr/include/c++/7 
QMAKE_LIBDIR += $$CUDA_DIR/lib
INCLUDEPATH += /usr/local/cuda/include
LIBS+=-L/usr/local/cuda/lib64 -L/usr/lib64 -lSDL2 -ldl -lpthread -lSDL2_image -lassimp -lcudart -lcuda
CUDA_LIBS += -L/usr/local/cuda/lib64 -L/usr/lib64 -lcudart -lcuda -lSDL2  
cuda.commands = nvcc -m64 -arch=$$CUDA_ARCH -c $$CUDA_SRC -o ${QMAKE_FILE_BASE}.o -lcuda -lcudart -lSDL2  
cuda.dependency_type = TYPE_C
cuda.input = CUDA_SRC
cuda.output = ${QMAKE_FILE_BASE}.o
QMAKE_EXTRA_COMPILERS+= cuda 
QMAKE_CXXFLAGS += -std=c++17 
}

# Input
HEADERS += "Form Files/test.h" \
	includes/*.h
FORMS += "Form Files/test.ui"
UI_DIR += "Form Files/" 
SOURCES += sources/*.cpp

RESOURCES += Ressources/Resource.qrc
