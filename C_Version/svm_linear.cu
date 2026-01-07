#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <assert.h>
#include "svm.hpp"

using namespace std;

#define MIN(a, b) ((a) > (b) ? (b) : (a))

// Macros to configure kernel parameters
#ifndef RUN_MODE
#define RUN_MODE THREAD_PARALLEL
#endif

#ifndef DIM
#define DIM 2
#endif

#ifndef THREADS_MAX
#define THREADS_MAX 32      // Maximum number of threads for THREAD_PARALLEL mode
#endif

#ifndef RANDOM
#define RANDOM TRUE        // Default to using random input data
#endif

#ifndef NUM_INPUTS
#define NUM_INPUTS 10000000 // Total number of input vectors (MAX: 10000000)
#endif

#ifndef BLOCK_THREADS
#define BLOCK_THREADS MIN(THREADS_MAX, 8)     // Threads per block in BLOCK_PARALLEL mode
#endif

#ifndef THREAD_THREADS
#define THREAD_THREADS MIN(THREADS_MAX, NUM_INPUTS) // Number of threads per block for THREAD_PARALLEL
#endif

// Execution modes for GPU
enum GpuMode { SERIAL, BLOCK_PARALLEL, THREAD_PARALLEL };

// ===================== CPU Implementation ============================
// CPU implementation: Computes SVM dot product in 2D using hardcoded weights for each input vector
// General CPU SVM implementation for arbitrary dimensions
void svm_inference_batched_cpu_general(
  float* weights,
  float* input_vectors,
  float* outputs,
  int num_inputs,
  int dim,
  float bias
) {
  for (int i = 0; i < num_inputs; ++i) {
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d)
      dot += weights[d] * input_vectors[i * dim + d];
    outputs[i] = transfer(dot + bias);
  }
}

// Specialized CPU SVM implementation for 2D input vectors
void svm_inference_batched_cpu_2d(
  float* weights,
  float input_vectors[][2],
  float* outputs,
  int num_inputs,
  float bias
) {
  // Specialized version for 2D input vectors with hardcoded dot product
  for (int i = 0; i < num_inputs; ++i) {
    float dot = weights[0] * input_vectors[i][0] + weights[1] * input_vectors[i][1];
    outputs[i] = transfer(dot + bias); // Apply transfer function (e.g., step or sigmoid)
  }
}

// ===================== GPU BLOCK_PARALLEL Kernel ============================
// GPU BLOCK_PARALLEL kernel: Each block handles one input vector and reduces dot product in parallel using shared memory
__global__ void svm_inference_batched_kernel(const float* weights, const float* input_vectors,
                                  float* outputs, int num_inputs, int dim, float bias) {
  int input_idx = blockIdx.x;      // Each block handles one input vector
  int tid = threadIdx.x;           // Thread index within the block
  extern __shared__ float shared[]; // Shared memory buffer for partial dot products

  float temp = 0.0f;
  // Each thread computes partial dot product over a chunk of dimensions
  // Each thread processes elements spaced blockDim.x apart to cover the entire input vector in a strided manner
  for (int i = tid; i < dim; i += blockDim.x) {
    float x = input_vectors[input_idx * dim + i];
    float w = weights[i];
    temp += x * w;
  }
  shared[tid] = temp;              // Store partial result in shared memory
  __syncthreads();                 // Wait for all threads

  // Reduce partial sums using parallel reduction
  // Perform parallel reduction: at each step, we halve the number of active threads
  // Conceptually, this builds a binary tree of additions where each thread at index 'i'
  // combines its partial sum with the sum from a partner 'i + stride'. By halving 'stride'
  // each iteration, the algorithm ensures that values propagate up to thread 0 in log2(threads) steps.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();               // Wait after each reduction step
  }

  // Thread 0 writes final result
  if (tid == 0)
    outputs[input_idx] = transfer(shared[0] + bias);
}

// ===================== GPU THREAD_PARALLEL Kernel ============================
__constant__ float d_const_weights[2]; // Constant memory for weights (2D only)

// GPU THREAD_PARALLEL kernel: Each thread processes one input vector using constant-memory weights for 2D classification
// GPU kernel for thread-parallel inference on arbitrary dimensions using global memory
// Each thread computes the dot product between one input vector and the full weight vector
// This allows the kernel to scale to inputs with any number of dimensions (dim > 2)
__global__ void svm_inference_threaded_kernel_general(
  const float* d_const_weights,  // Weight vector passed as constant or global memory
  const float* input_vectors,    // Flattened 2D input array
  float* outputs,                // Output prediction scores
  int num_inputs,                // Number of input vectors
  int dim,                       // Dimension of each input vector
  float bias                    // Bias term to add after dot product
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Each thread handles one input vector
  if (idx >= num_inputs) return;                    // Thread bounds check

  float dot = 0.0f;
  // Compute the dot product between input vector and weights
  for (int d = 0; d < dim; ++d)
    dot += d_const_weights[d] * input_vectors[idx * dim + d];

  // Apply activation function to final result and store output
  outputs[idx] = transfer(dot + bias);
}


// GPU kernel specialized for 2D input using constant memory
// Each thread computes a hardcoded 2D dot product using fast access from __constant__ memory
// This version is more efficient for low-dimensional problems like 2D binary classification
__global__ void svm_inference_threaded_kernel_2d(
  const float* input_vectors,    // Flattened 2D array of inputs
  float* outputs,                // Output array for results
  int num_inputs,                // Number of input samples
  float bias                     // Bias term to apply
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Each thread processes one input vector
  if (idx >= num_inputs) return;                    // Bounds check

  // Manually unroll the 2D dot product using known dimensions
  float x0 = input_vectors[idx * 2];
  float x1 = input_vectors[idx * 2 + 1];
  float dot = d_const_weights[0] * x0 + d_const_weights[1] * x1;

  // Apply transfer function and write result
  outputs[idx] = transfer(dot + bias);
}

// ===================== GPU Dispatcher ============================
// Host-side dispatcher: Chooses and launches the appropriate GPU kernel (BLOCK_PARALLEL or THREAD_PARALLEL) based on mode
void svm_inference_gpu_dispatch(const float* h_weights, const float* h_inputs,
                                float* h_outputs, int num_inputs, int dim, float bias, GpuMode mode) {
  float *d_weights = nullptr, *d_inputs = nullptr, *d_outputs = nullptr;
  // Allocate device memory
  cudaMalloc(&d_weights, dim * sizeof(float));
  cudaMalloc(&d_inputs, num_inputs * dim * sizeof(float));
  cudaMalloc(&d_outputs, num_inputs * sizeof(float));

  // Copy weights and inputs from host to device
  cudaMemcpy(d_weights, h_weights, dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inputs, h_inputs, num_inputs * dim * sizeof(float), cudaMemcpyHostToDevice);

  // Launch appropriate kernel depending on mode
  if (mode == BLOCK_PARALLEL) {
    begin_roi(); // Optional timing macro
    svm_inference_batched_kernel<<<num_inputs, BLOCK_THREADS, BLOCK_THREADS * sizeof(float)>>> (
      d_weights, d_inputs, d_outputs, num_inputs, dim, bias);
    cudaDeviceSynchronize(); // Wait for kernel to finish
    end_roi("GPU BLOCK PARALLEL");
  } else if (mode == THREAD_PARALLEL) {
    // Use constant memory weights
    cudaMemcpyToSymbol(d_const_weights, h_weights, 2 * sizeof(float));
    int threads = THREAD_THREADS;
    int blocks = (num_inputs + threads - 1) / threads; // Ceiling division
    begin_roi();
    #if DIM == 2
    svm_inference_threaded_kernel_2d<<<blocks, threads>>>(d_inputs, d_outputs, num_inputs, bias);
    #else
    svm_inference_threaded_kernel_general<<<blocks, threads>>>(d_weights, d_inputs, d_outputs, num_inputs, dim, bias);
    #endif
    cudaDeviceSynchronize();
    end_roi("GPU THREAD PARALLEL");
  }

  // Copy outputs from device to host
  cudaMemcpy(h_outputs, d_outputs, num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_weights); cudaFree(d_inputs); cudaFree(d_outputs);
}

// ===================== MAIN ============================
int main() {
  const int num_inputs = NUM_INPUTS;      // Total input samples

  #if DIM == 2
  const int dim = 2;
  float weights[] = {1.796463f, -2.053094f};
  float bias = -3.438080f;
#else
  const int dim = DIM;
  float* weights = new float[dim];
  float bias = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random bias in [-1,1]
  for (int i = 0; i < dim; ++i) {
    weights[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random weights in [-1,1]
  }
#endif                // SVM bias term

  // Allocate and generate random test data in both 2D and flat format
  float (*test_data_2d)[DIM] = new float[num_inputs][DIM];
  float* test_data_flat = new float[num_inputs * dim];
  for (int i = 0; i < num_inputs; ++i) {
    for(int j = 0; j < dim; ++j) {
      test_data_2d[i][j] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
      test_data_flat[i * dim + j] = test_data_2d[i][j];  
    }
  }

  // Output buffers for each implementation
  float* cpu_batch_out = new float[num_inputs];
  float* gpu_out = new float[num_inputs];

  // Run CPU implementation
  begin_roi();
  #if DIM == 2
  svm_inference_batched_cpu_2d(weights, test_data_2d, cpu_batch_out, num_inputs, bias);
  #else
  svm_inference_batched_cpu_general(weights, test_data_flat, cpu_batch_out, num_inputs, dim, bias);
  #endif
  end_roi("CPU BATCHED");

  // Run selected GPU version based on RUN_MODE
  svm_inference_gpu_dispatch(weights, test_data_flat, gpu_out, num_inputs, dim, bias, RUN_MODE);

  // Compare CPU and GPU outputs
  bool all_match = true;
  for (int i = 0; i < num_inputs; ++i) {
    if (fabs(cpu_batch_out[i] - gpu_out[i]) > 0.001f) {
      std::cerr << "Mismatch at " << i << " cpu_batch=" << cpu_batch_out[i]
                << ", gpu=" << gpu_out[i] << endl;
      all_match = false;
    }
  }

  if (all_match)
    std::cout << "All outputs match across all implementations!\n";

  // Clean up heap-allocated memory
  delete[] test_data_flat;
  delete[] test_data_2d;

  delete[] cpu_batch_out;
  delete[] gpu_out;

  #if DIM > 2
  delete[] weights;
  #endif

  return 0;
}
