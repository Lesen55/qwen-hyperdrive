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
#define CUdeviceptr hipDeviceptr_t

// ---- runtime API values
#define cudaSuccess hipSuccess
#define cudaStreamNonBlocking hipStreamNonBlocking
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
#define cudaFuncSetAttribute hipFuncSetAttribute

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
