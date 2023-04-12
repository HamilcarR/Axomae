TEMPLATE = app
TARGET = Axomae

QT += widgets\
      gui\
      openglwidgets\

CONFIG += debug \
          warn_on \	


########################################################################################
# Input
HEADERS += "Form Files/test.h" \
	includes/*.h
OBJECTS_DIR=generated_files
DESTDIR=bin
FORMS += "Form Files/test.ui"
UI_DIR += "Form Files/" 
SOURCES += sources/*.cpp
RESOURCES += Ressources/Resource.qrc

########################################################################################
#Configure these env variables before compiling#
CUDA_ARCH = sm_75
CUDA_SRC += kernels/*.cu

#Linux rules#
linux:{
QMAKE_CC = gcc
QMAKE_CXX = g++
QMAKE_LINK = g++
INCLUDEPATH +=	/usr/include/SDL2 \
		/usr/local/cuda/include \
		/usr/include/glm \
		/usr/include/GL\

#QMAKE_DEFAULT_INCDIRS += -I/usr/include/c++/13 
QMAKE_LIBDIR += $$CUDA_DIR/lib
LIBS+=-L/usr/local/cuda/lib64 -L/usr/lib64 -lSDL2 -ldl -lpthread -lSDL2_image -lassimp -lcudart -lcuda -lGLEW -lGLU -lGL
CUDA_LIBS += -L/usr/local/cuda/lib64 -L/usr/lib64 -lcudart -lcuda -lSDL2  
cuda.commands = nvcc --expt-relaxed-constexpr --compiler-bindir /usr/bin/g++-13 -m64 -arch=$$CUDA_ARCH -std=c++17 --device-debug -c $$CUDA_SRC -o ${QMAKE_FILE_BASE}.o -lcuda -lcudart -lSDL2  
cuda.dependency_type = TYPE_C
cuda.input = CUDA_SRC
cuda.output = ${QMAKE_FILE_BASE}.o
QMAKE_EXTRA_COMPILERS+= cuda 
QMAKE_CXXFLAGS += -std=c++17 -g -Wall -pedantic -Wno-unused 
}

########################################################################################

