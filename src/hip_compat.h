// HIP compatibility layer (Windows port, QE_HIP builds only).
//
// Lets the CUDA sources (model.h / model.cpp / kernels.h / kernels.cu) compile
// against the HIP runtime for AMD GPUs: host-side CUDA API names are mapped 1:1
// to their HIP equivalents, and cuBLAS to hipBLAS. Device-side PTX (cp.async,
// ldmatrix, mma.sync) is NOT mapped here; the tensor-core kernels that use it
// are disabled on HIP in kernels.cu (their *_supported() gates return false and
// the engine's portable paths take over).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

// ---- runtime API types
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
// CUDA's CUdeviceptr is an integer; HIP's hipDeviceptr_t is void* on Windows,
// which breaks address arithmetic in the model code. Keep the integer type and
// convert at the stream-value call sites below.
#define CUdeviceptr unsigned long long

// ---- runtime API values
#define cudaSuccess hipSuccess
#define CUDA_SUCCESS hipSuccess
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaEventDisableTiming hipEventDisableTiming
#define CUstream hipStream_t
#define CU_STREAM_WRITE_VALUE_DEFAULT 0u  // hipStreamWriteValueDefault; not exported by the 7.1 Windows headers
#define CU_STREAM_WAIT_VALUE_GEQ hipStreamWaitValueGte
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaHostAllocDefault 0
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// ---- runtime API functions
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cuStreamWriteValue32 qeStreamWriteValue32
#define cuStreamWaitValue32 qeStreamWaitValue32
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemsetAsync hipMemsetAsync
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaMemGetInfo hipMemGetInfo
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaFuncSetAttribute qeFuncSetAttribute

// ---- wrappers whose HIP signatures differ from CUDA's
// Kernel function references (incl. dependent template instantiations) do not
// implicitly convert to const void* under HIP clang; cast explicitly.
template <typename F>
static inline hipError_t qeFuncSetAttribute(F func, hipFuncAttribute attr, int value) {
  return hipFuncSetAttribute((const void*)func, attr, value);
}
// Stream write/wait value take an integer address here (see CUdeviceptr note).
static inline hipError_t qeStreamWriteValue32(hipStream_t s, unsigned long long addr, unsigned int v,
                                              unsigned int flags) {
  return hipStreamWriteValue32(s, (hipDeviceptr_t)(uintptr_t)addr, v, flags);
}
static inline hipError_t qeStreamWaitValue32(hipStream_t s, unsigned long long addr, unsigned int v,
                                             unsigned int flags) {
  return hipStreamWaitValue32(s, (hipDeviceptr_t)(uintptr_t)addr, v, flags);
}

// ---- warp shuffle mask width
// CUDA code passes 32-bit masks (0xffffffff); HIP's __shfl*_sync templates
// static-assert a 64-bit mask. Widen at the call sites via macros (the real
// builtins are used inside the wrappers, which are declared before the macros).
template <typename T>
static __device__ __forceinline__ T qe_shfl_xor_sync(unsigned int mask, T var, int lane) {
  return __shfl_xor_sync((unsigned long long)mask, var, lane);
}
template <typename T>
static __device__ __forceinline__ T qe_shfl_sync(unsigned int mask, T var, int lane) {
  return __shfl_sync((unsigned long long)mask, var, lane);
}
#define __shfl_xor_sync(mask, var, lane) qe_shfl_xor_sync(mask, var, lane)
#define __shfl_sync(mask, var, lane) qe_shfl_sync(mask, var, lane)

// ---- device intrinsics HIP does not ship
// __dp4a: 4-way signed/unsigned byte dot product plus accumulator.
// gfx1100 has no v_dot4_i32_i8 (that is the CDNA encoding); its native int8
// dot4 is v_dot4_i32_iu8 via __builtin_amdgcn_sudot4 (first operand signed,
// second unsigned), which needs no extra target feature. (The sdot4 builtin
// is feature-gated on dot1-insts, and opting into that feature per-function
// or TU-wide broke kernel registration at runtime -- do not go back there.)
// Signed x signed and unsigned x unsigned are recovered exactly by biasing
// one operand's bytes by +128 (xor 0x80) and correcting with the other
// operand's byte sum, also computed with sudot4: 2 dot4 + a few ops instead
// of the ~12-instruction scalar expansion.
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
  const int biased = __builtin_amdgcn_sudot4(a, b ^ (int)0x80808080, c);
  const int sum_a = __builtin_amdgcn_sudot4(a, 0x01010101, 0);
  return biased - (sum_a << 7);
}
static __device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c) {
  const int biased = __builtin_amdgcn_sudot4((int)(a ^ 0x80808080u), (int)b, (int)c);
  const int sum_b = __builtin_amdgcn_sudot4(0x01010101, (int)b, 0);
  return (unsigned int)(biased + (sum_b << 7));
}
// __vsub4: per-byte wrapping subtraction (CUDA __nvvm vsub4).
static __device__ __forceinline__ unsigned int __vsub4(unsigned int a, unsigned int b) {
  unsigned int r = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const unsigned int x = (a >> (8 * i)) & 0xffu, y = (b >> (8 * i)) & 0xffu;
    r |= ((x - y) & 0xffu) << (8 * i);
  }
  return r;
}

// ---- cuBLAS -> hipBLAS
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
