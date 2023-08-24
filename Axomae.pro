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

FORMS += "Form Files/test.ui"
UI_DIR += "Form Files/" 
SOURCES += sources/*.cpp\
		   tests/*.cpp
RESOURCES += Ressources/Resource.qrc
TESTS += tests/*.cpp
TESTDATA += /usr/local/include/gtest
########################################################################################
#Configure these env variables before compiling#
DESTDIR += bin/
#Linux rules#
linux:{
CUDA_DIR = /usr/local/cuda
QMAKE_CC = gcc-12
QMAKE_CXX = g++-12
QMAKE_LINK = g++-12
INCLUDEPATH +=	/usr/include/SDL2 \
		/usr/local/cuda/include \
		/usr/include/glm \
		/usr/include/GL\
		vendor/*.h \

#QMAKE_DEFAULT_INCDIRS += -I/usr/include/c++/12
QMAKE_CXXFLAGS += -std=c++17 -g -pg -Wall -pedantic -Wno-unused -O3 
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS+=-L/usr/local/cuda/lib64 -L/usr/lib64 -lSDL2 -ldl -lpthread -lSDL2_image -lassimp -lcudart -lcudadevrt -lcuda -lGLEW -lGLU -lglut -lGL -lgtest
CUDA_LIBS += -L/usr/local/cuda/lib64 -L/usr/lib64 -lcudart -lcuda -lcudadevrt -lSDL2 
CUDA_ARCH = sm_75
CUDA_SRC += kernels/*.cu	
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ') 
NVCCFLAGS = --expt-relaxed-constexpr --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v --device-debug 
cudaIntr.input = CUDA_SRC
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
 
cudaIntr.commands = nvcc --compiler-bindir /usr/bin/g++-12 -m64 -g -G -arch=$$CUDA_ARCH -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr
# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

}

########################################################################################

