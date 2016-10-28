CC=g++ 
CFLAGS=-std=c++14 -Wall -pedantic -pthread -g
CLIBS=-lSDL2 -lSDL2_image
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)
TARGET=program

all : $(TARGET)
	mv $^ ../

$(TARGET) : $(OBJ)
	@$(CC) -o $@ $^ $(CLIBS)

%.o:%.cpp
	@$(CC) -c $< $(CFLAGS) -o $@


.PHONY: clean

clean:
	rm -f *.o
