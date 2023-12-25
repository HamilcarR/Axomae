TEMPLATE = app
TARGET = Tests
CONFIG +=   debug \
            warn_on 
QT += widgets\
      gui\
      openglwidgets\
      testlib
	  

SRC = $$PWD/../sources
INCLUDEPATH +=	/usr/include/SDL2 \
		        /usr/local/cuda/include \
		        /usr/include/glm \
		        /usr/include/GL\
		        $$PWD/../vendor/stb

HEADERS += $$PWD/Test.h
SOURCES += $$PWD/Test.cpp
TESTDATA += /usr/local/include/gtest
MOC_DIR = $$PWD/moc



include($$SRC/modules.pri)
include($$SRC/database/test/test_database.pri)
include($$SRC/mesh/test/test_mesh.pri)
include($$SRC/scene/test/test_scene.pri)
include($$SRC/processing/test/test_processing.pri)




########################################################################################
#Configure these env variables before compiling#
DESTDIR += bin/
QMAKE_CC = gcc-12
QMAKE_CXX = g++-12
QMAKE_LINK = g++-12
OBJECTS_DIR = $$PWD/../"generated files"
CUDA_DIR = /usr/local/cuda

QMAKE_CXXFLAGS += -std=c++17 -g -pg -Wall -pedantic -Wno-unused 
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS+=-L/usr/local/cuda/lib64 -L/usr/lib64 -lSDL2 -ldl -lpthread -lSDL2_image -lassimp -lcudart -lcudadevrt -lcuda -lGLEW -lGLU -lglut -lGL -lgtest
CUDA_LIBS += -L/usr/local/cuda/lib64 -L/usr/lib64 -lcudart -lcuda -lcudadevrt -lSDL2 
CUDA_ARCH = sm_75
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
CUDA_PROFILE_FLAG = -g -G -lineinfo 
CUDA_DEBUG_FLAG = -g -G --device-debug
CUDA_OPTI_FLAG = -Xptxas -O3 -use_fast_math
NVCCFLAGS =  $$CUDA_DEBUG_FLAG --threads 8  
cudaIntr.input = CUDA_SRC
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
 
cudaIntr.commands = nvcc --compiler-bindir /usr/bin/g++-12 -m64 -arch=$$CUDA_ARCH -dc $$NVCCFLAGS $$CUDA_INC ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr
# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -arch=$$CUDA_ARCH -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = nvcc -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda



########################################################################################







