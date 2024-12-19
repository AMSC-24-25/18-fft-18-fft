# Fast Fourier Transform (FFT) Repository

## Overview

This repository contains implementations of the **Fast Fourier Transform (FFT)** algorithm in both sequential and parallel versions. FFT is a fundamental algorithm in digital signal processing, used to compute the Discrete Fourier Transform (DFT) and its inverse efficiently.

The repository includes the following implementations:

1. **Sequential FFT**:
   - **Iterative**: A non-recursive implementation in C++.
   - **Recursive**: A divide-and-conquer implementation in C++.

2. **Parallel FFT**:
   - **CUDA Implementation**: A GPU-accelerated iterative FFT for high-performance computation.
   - **Multithreaded C++ Implementation**: A CPU-parallel iterative FFT using OpenMP directives.

---

## File Structure
```plaintext
/src
├── Cooley-Tukey.cpp            # Sequential iterative and recursive FFT implementation as object
├── Cooley-Tukey-parallel.cpp   # Parallel iterative FFT implementation   as object
├── cuda.cu                     # CUDA-based parallel FFT implementation
├── main.cpp                    # Main
├── Makefile                    # Makefile for compile the entire project
/include
├── Cooley-Tukey.hpp            # Sequential iterative and recursive FFT header
├── Cooley-Tukey-parallel.hpp   # Parallel iterative FFT header
```
## How to run

1. Go to /src folder
2. run ```make``` on th command line
3. A /bin folder will be created, here you can run the main: ```./main```


## Implementation
The execution of the main() compares the execution of sequential and parallel FFT algorithm. 
Two objects are created respectively to solve the FFT. Then the computational time is measured and compares the 2 results.
