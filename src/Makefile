BIN_DIR=../bin
PROGRAM1=$(BIN_DIR)/cooley_tukey-co
INCLUDE_DIR=../include
INCLUDES=$(INCLUDE_DIR)/Cooley-Tukey.hpp $(INCLUDE_DIR)/Cooley-Tukey-parallel.hpp

SOURCES=Cooley-Tukey.cpp Cooley-Tukey-parallel.cpp

CC=g++
CFLAGS=-O3 -fopenmp
LDFLAGS=-fopenmp

all: $(PROGRAM1) $(PROGRAM2)

$(PROGRAM1): $(SOURCES)
	$(CC) $(CFLAGS) $(VERBOSE) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
