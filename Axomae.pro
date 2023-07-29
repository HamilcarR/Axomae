TEMPLATE = app
TARGET = Axomae

QT += widgets\
      gui\
      openglwidgets\

CONFIG += debug \
          warn_on \
		  testcase	



########################################################################################
# Input
HEADERS += "Form Files/test.h" \
	includes/*.h
OBJECTS_DIR=generated_files
DESTDIR=bin
FORMS += "Form Files/test.ui"
UI_DIR += "Form Files/" 
SOURCES += sources/*.cpp\
		   tests/*.cpp
RESOURCES += Ressources/Resource.qrc
TESTS += tests/*.cpp
TESTDATA += /usr/local/include/gtest
########################################################################################
#Configure these env variables before compiling#
CUDA_ARCH = sm_75
CUDA_SRC += kernels/*.cu

#Linux rules#
linux:{
QMAKE_CC = gcc-12
QMAKE_CXX = g++-12
QMAKE_LINK = g++-12
INCLUDEPATH +=	/usr/include/SDL2 \
		/usr/local/cuda/include \
		/usr/include/glm \
		/usr/include/GL\
		vendor/*.h \

#QMAKE_DEFAULT_INCDIRS += -I/usr/include/c++/12
QMAKE_LIBDIR += $$CUDA_DIR/lib
LIBS+=-L/usr/local/cuda/lib64 -L/usr/lib64 -lSDL2 -ldl -lpthread -lSDL2_image -lassimp -lcudart -lcuda -lGLEW -lGLU -lglut -lGL -lgtest
CUDA_LIBS += -L/usr/local/cuda/lib64 -L/usr/lib64 -lcudart -lcuda -lSDL2  
cuda.commands = nvcc --expt-relaxed-constexpr --compiler-bindir /usr/bin/g++-12 -m64 -arch=$$CUDA_ARCH -std=c++17 --device-debug -c $$CUDA_SRC -o ${QMAKE_FILE_BASE}.o -lcuda -lcudart -lSDL2  
cuda.dependency_type = TYPE_C
cuda.input = CUDA_SRC
cuda.output = ${QMAKE_FILE_BASE}.o
QMAKE_EXTRA_COMPILERS+= cuda 
QMAKE_CXXFLAGS += -std=c++17 -g -pg -Wall -pedantic -Wno-unused 
}

########################################################################################

