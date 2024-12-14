BIN_DIR=../bin
PROGRAM1=$(BIN_DIR)/cooley_tukey-co

CC=g++
CFLAGS=-O3 -fopenmp
LDFLAGS=-fopenmp

all: $(PROGRAM1) $(PROGRAM2)

$(PROGRAM1): Cooley-Tukey.cpp
	$(CC) $(CFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
