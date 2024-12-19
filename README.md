Here’s your updated content formatted professionally for GitHub, retaining the structure and details you provided:

---

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
├── Cooley-Tukey.cpp            # Sequential iterative and recursive FFT implementation
├── Cooley-Tukey-parallel.cpp   # Parallel iterative FFT implementation
├── cuda.cu                     # CUDA-based parallel FFT implementation
├── main.cpp                    # Main driver program
├── Makefile                    # Makefile for building the entire project
/include
├── Cooley-Tukey.hpp            # Header for sequential FFT
├── Cooley-Tukey-parallel.hpp   # Header for parallel FFT
```

---

## How to Run

1. Navigate to the `/src` folder:
   ```bash
   cd src
   ```
2. Build the project using the `Makefile`:
   ```bash
   make
   ```
3. After compilation, a `/bin` folder will be created. Run the main executable:
   ```bash
   ./main
   ```

---

## Implementation Details

The `main()` function compares the execution of sequential and parallel FFT algorithms.  
- Two objects are created to execute the FFT using different approaches.
- The computational time for both methods is measured and their performance is compared.

---

## CUDA Overview

### Kernel Launch Configuration

CUDA kernels are launched using a grid of thread blocks. The configuration is defined by two parameters:
- **`block_dim`**: Specifies the number of threads in a block.
- **`grid_size`**: Specifies the number of blocks in the grid.

#### Example Kernel Launch
```cpp
parallel_fft<<<dimGrid, dimBlock>>>(a, x, t, log_n);
```

### Thread Data Flow

The data flow between threads during FFT computation is illustrated below:

![FFT Data Flow](https://github.com/user-attachments/assets/228cec06-f8d6-4a67-b224-23871296f993)

Threads exchange data as needed to perform efficient FFT computations while minimizing global memory access.

### Thread Computation

Each thread computes part of the FFT independently. The process is depicted below:

![Thread Computation](https://github.com/user-attachments/assets/65ecc3f7-051f-4c67-93bb-740e85ce6cc6)

---
