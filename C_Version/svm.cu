#include <ctime>
#include <cstdlib>
#include <iostream>
#include <string>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "svm.hpp"

using namespace std;

// --------------------- SVM MODEL CONSTANTS ----------------------------

// constexpr float SUPPORT_VECTORS[NUM_SV][DIM] = {
//   {5.0000000000, 3.2000000000},
//   {4.9000000000, 3.1000000000},
//   {5.0000000000, 3.0000000000},
//   {5.5000000000, 3.5000000000},
//   {4.5000000000, 2.3000000000},
//   {4.9000000000, 3.0000000000},
//   {5.4000000000, 3.4000000000},
//   {5.8000000000, 4.0000000000},
//   {6.0000000000, 3.4000000000},
//   {5.6000000000, 3.0000000000},
//   {5.7000000000, 3.0000000000},
//   {5.6000000000, 2.9000000000},
//   {5.1000000000, 2.5000000000},
//   {4.9000000000, 2.4000000000},
//   {5.2000000000, 2.7000000000},
//   {5.4000000000, 3.0000000000}
// };

// constexpr float ALPHAS[NUM_SV] = {
//   0.4693142383, 1.0000000000, 1.0000000000, 1.0000000000,
//   1.0000000000, 1.0000000000, 1.0000000000, 0.8011613539,
//   1.0000000000, 1.0000000000, 1.0000000000, 0.2704755922,
//   1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000
// };

// constexpr int LABELS[NUM_SV] = {
//   -1, -1, -1, -1,
//   -1, 1, 1, 1,
//   -1, -1, -1, -1,
//   -1, -1, -1, 1
// };

// constexpr float BIAS = -4.9957748767f;

// ==============================
// Configurable Parameters
// ==============================
#ifndef SVM_DIM
#define SVM_DIM 2
#endif

#ifndef SVM_NUM_SV
#define SVM_NUM_SV 16
#endif

#ifndef NUM_TEST
#define NUM_TEST 10000000
#endif

#ifndef SVM_GPU_MODE
#define SVM_GPU_MODE THREAD  // Options: SINGLE, BLOCK, THREAD
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256       // Threads per block for SINGLE and BLOCK kernels
#endif

#ifndef THREAD_BLOCK_SIZE
#define THREAD_BLOCK_SIZE 128  // Threads per block for THREAD mode
#endif

// --------------------- SVM FUNCTIONS ----------------------------

// CPU reference implementation of SVM inference over a batch of input vectors.
// It computes the decision function sum over support vectors, applying a linear kernel.
// Used as a baseline for correctness verification.
void svm_inference_batched_cpu(
  float* support_vectors, float* alphas, float* labels,
  float* input_vectors, float* outputs,
  int num_inputs, int num_sv, int dim, float bias
) {
  for (int i = 0; i < num_inputs; i++) {
    float sum = 0.0f;
    for (int sv = 0; sv < num_sv; sv++) {
      float dot = 0.0f;
      for (int d = 0; d < dim; d++) {
        dot += support_vectors[sv * dim + d] * input_vectors[i * dim + d];
      }
      sum += alphas[sv] * labels[sv] * dot; // Contribution of this support vector
    }
    outputs[i] = transfer(sum + bias); // Apply bias and transform the decision value
  }
}

__global__
void svm_inference_kernel(const float* support_vectors, const float* alphas, const float* labels,
                          const float* input_vector, float* output, int num_sv, int dim, float bias) {
  // This kernel handles inference for a **single input vector**.
  // Each thread computes the dot product between the input vector and a support vector.
  // Then it contributes to the final classification output using shared memory reduction.

  __shared__ float partial_sum[256]; // Shared memory buffer to store per-thread partial results.

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid; // Global support vector index.

  float sum = 0.0f;
  if (idx < num_sv) {
    float dot = 0.0f;
    // Compute dot product between the support vector and the input vector.
    for (int d = 0; d < dim; d++) {
      dot += support_vectors[idx * dim + d] * input_vector[d];
    }
    // Each thread computes one term of the decision function.
    sum = alphas[idx] * labels[idx] * dot;
  }

  partial_sum[tid] = sum; // Write partial result to shared memory.
  __syncthreads();

  // Tree-based parallel reduction to sum all partial results within the block.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      partial_sum[tid] += partial_sum[tid + stride];
    __syncthreads();
  }

  // Final result is written atomically to the global output variable.
  if (tid == 0) {
    atomicAdd(output, partial_sum[0]);
  }
}

__global__
void svm_inference_batched_kernel(const float* support_vectors, const float* alphas, const float* labels,
                                  const float* input_vectors, float* outputs,
                                  int num_inputs, int num_sv, int dim, float bias) {
    // This kernel handles inference for **multiple input vectors** in a batched manner.
    // Each block processes one input vector (indexed by blockIdx.x).
    // Each thread computes the contribution of a single support vector.
    // A shared memory reduction sums up the results.

    extern __shared__ float partial_sum[]; // Dynamically allocated shared memory to store per-thread partial results.

    int tid = threadIdx.x;
    int input_idx = blockIdx.x;     // Each block processes a unique input vector.
    int sv_idx = threadIdx.x;       // Each thread corresponds to a support vector.

    float sum = 0.0f;
    if (sv_idx < num_sv && input_idx < num_inputs) {
        float dot = 0.0f;
        // Compute dot product between current input vector and the assigned support vector.
        for (int d = 0; d < dim; d++) {
            float x_val = input_vectors[input_idx * dim + d];
            float sv_val = support_vectors[sv_idx * dim + d];
            dot += sv_val * x_val;
        }
        // Each thread computes its contribution to the classification result.
        sum = alphas[sv_idx] * labels[sv_idx] * dot;
    }

    partial_sum[tid] = sum; // Store result in shared memory.
    __syncthreads();

    // Perform reduction within the block to sum all thread contributions.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial_sum[tid] += partial_sum[tid + stride];
        __syncthreads();
    }

    // Thread 0 of each block finalizes the classification for its input vector.
    if (tid == 0) {
        float result = partial_sum[0] + bias;
        outputs[input_idx] = transfer(result);
    }
}

// Thread-parallel kernel version where each thread handles a single input vector independently
// Now enhanced with shared memory optimization for support vectors
__global__
void svm_inference_thread_parallel_kernel(const float* __restrict__ support_vectors,
                                           const float* __restrict__ alphas,
                                           const float* __restrict__ labels,
                                           const float* __restrict__ input_vectors,
                                           float* __restrict__ outputs,
                                           int num_inputs, int num_sv, int dim, float bias) {

    // Each thread processes one input vector from the batch
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_idx >= num_inputs) return;

    // Declare shared memory for storing support vectors
    extern __shared__ float shared_sv[];  // Size = num_sv * dim floats

    // Cooperative loading of support_vectors into shared memory
    int threads = blockDim.x;
    int total_elems = num_sv * dim;

    for (int idx = threadIdx.x; idx < total_elems; idx+=threads) {
        shared_sv[idx] = support_vectors[idx];
    }
    __syncthreads(); // Ensure shared memory is fully populated before use

    // Precompute input base index
    int input_base = input_idx * dim;

    float sum = 0.0f;
    for (int sv = 0; sv < num_sv; sv++) {
      // Start computing dot product between input and this support vector
      float dot = 0.0f;
      int sv_base = sv * dim;

      // The inner loop computes the dot product between the input vector and a support vector
      // using values from global memory (input) and shared memory (support vectors)
      #pragma unroll
      for (int d = 0; d < dim; d++) {
          float x_val = input_vectors[input_base + d];
          float sv_val = shared_sv[sv_base + d];
          dot += sv_val * x_val;
      }
      sum += alphas[sv] * labels[sv] * dot;
    }

    // Final classification score after summing all contributions is transformed and stored
    outputs[input_idx] = transfer(sum + bias);
}

// Mode switcher for kernel selection
enum GpuMode { SINGLE, BLOCK, THREAD };

// ==============================================================
//  GPU Launcher: Chooses and runs the appropriate SVM kernel
// ==============================================================
void svm_inference_gpu_launcher(const float* h_sv, const float* h_alpha, const float* h_label,
                                const float* h_input, float* h_output,
                                int num_inputs, int num_sv, int dim, float bias, GpuMode mode) {
    // Allocate device memory for support vectors, alphas, labels, inputs, outputs
    float *d_sv, *d_alpha, *d_label, *d_input, *d_output;
    cudaMalloc(&d_sv, num_sv * dim * sizeof(float));
    cudaMalloc(&d_alpha, num_sv * sizeof(float));
    cudaMalloc(&d_label, num_sv * sizeof(float));
    cudaMalloc(&d_input, num_inputs * dim * sizeof(float));
    cudaMalloc(&d_output, num_inputs * sizeof(float));

    // Copy model parameters and input data from host to device
    cudaMemcpy(d_sv, h_sv, num_sv * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, num_sv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, h_label, num_sv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, num_inputs * dim * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = BLOCK_SIZE;  // Standard block size used in all kernels

    // ------------------------
    // SINGLE: One input vector processed with atomic add
    // ------------------------
    if (mode == SINGLE) {
        cudaMemset(d_output, 0, sizeof(float));  // Zero out output memory for atomic accumulation
        int grid_size = (num_sv + block_size - 1) / block_size;

        begin_roi();
        svm_inference_kernel<<<grid_size, block_size>>>(d_sv, d_alpha, d_label, d_input, d_output, num_sv, dim, bias);
        cudaDeviceSynchronize();
        end_roi("GPU SINGLE");

        // Copy result for single vector output back to host
        float result;
        cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        h_output[0] = transfer(result + bias);

    // ------------------------
    // BLOCK: Each block handles one input vector using intra-block reduction
    // ------------------------
    } else if (mode == BLOCK) {
        int shared_mem = block_size * sizeof(float);  // Shared memory size per block

        begin_roi();
        svm_inference_batched_kernel<<<num_inputs, block_size, shared_mem>>>(
            d_sv, d_alpha, d_label, d_input, d_output,
            num_inputs, num_sv, dim, bias);
        cudaDeviceSynchronize();
        end_roi("GPU BLOCK");

        // Copy all predictions back to host
        cudaMemcpy(h_output, d_output, num_inputs * sizeof(float), cudaMemcpyDeviceToHost);

    // ------------------------
    // THREAD: Each thread processes one input using shared memory caching of support vectors
    // ------------------------
    } else if (mode == THREAD) {
        int thread_block_size = THREAD_BLOCK_SIZE;  // Threads per block
        int thread_grid_size = (num_inputs + thread_block_size - 1) / thread_block_size;
        int shared_mem_size = num_sv * dim * sizeof(float);  // Shared memory to hold all support vectors

        begin_roi();
        svm_inference_thread_parallel_kernel<<<thread_grid_size, thread_block_size, shared_mem_size>>>(
            d_sv, d_alpha, d_label, d_input, d_output,
            num_inputs, num_sv, dim, bias);
        cudaDeviceSynchronize();
        end_roi("GPU THREAD");

        // Copy all predictions back to host
        cudaMemcpy(h_output, d_output, num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Cleanup: free all allocated GPU memory
    cudaFree(d_sv); cudaFree(d_alpha); cudaFree(d_label);
    cudaFree(d_input); cudaFree(d_output);
}


// ==============================================================
//  MAIN: Runs CPU and GPU SVM inference and compares results
// ==============================================================
int main() {
    const int dim = SVM_DIM;
    const int num_sv = SVM_NUM_SV;
    const int num_inputs = NUM_TEST;

    // Random seed for reproducibility
    srand(42);

    // Allocate and initialize model parameters
    float* support_vectors = new float[num_sv * dim];
    float* alphas = new float[num_sv];
    float* labels = new float[num_sv];
    float bias;

    // Generate support vectors with values in [-2, 2]
    for (int i = 0; i < num_sv * dim; ++i) {
        support_vectors[i] = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
    }

    // Generate alpha values in [0, 1]
    for (int i = 0; i < num_sv; ++i) {
        alphas[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Random binary labels {-1, +1}
    for (int i = 0; i < num_sv; ++i) {
        labels[i] = (rand() % 2) ? 1.0f : -1.0f;
    }

    // Random bias in [-2, 2]
    bias = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;

    // Allocate and generate input vectors
    float* input_vectors = new float[num_inputs * dim];
    for (int i = 0; i < num_inputs * dim; ++i) {
        input_vectors[i] = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
    }

    // Allocate output buffers
    float* out_cpu = new float[num_inputs];
    float* out_gpu = new float[num_inputs];

    // Run CPU inference
    begin_roi();
    svm_inference_batched_cpu(support_vectors, alphas, labels,
                              input_vectors, out_cpu,
                              num_inputs, num_sv, dim, bias);
    end_roi("Batched CPU");

    // Select and run GPU inference
    GpuMode mode = SVM_GPU_MODE;
    svm_inference_gpu_launcher(support_vectors, alphas, labels,
                               input_vectors, out_gpu,
                               num_inputs, num_sv, dim, bias, mode);

    // Compare results
    bool all_match = true;
    for (int i = 0; i < num_inputs; ++i) {
        if (fabs(out_cpu[i] - out_gpu[i]) > 0.001f) {
            std::cerr << "Mismatch at index " << i << ": cpu=" << out_cpu[i]
                      << ", gpu=" << out_gpu[i] << "\n";
            all_match = false;
        }
    }

    if (all_match) {
        std::cout << "All outputs match across all implementations!\n";
    }

    // Free memory
    delete[] support_vectors;
    delete[] alphas;
    delete[] labels;
    delete[] input_vectors;
    delete[] out_cpu;
    delete[] out_gpu;

    return 0;
}
