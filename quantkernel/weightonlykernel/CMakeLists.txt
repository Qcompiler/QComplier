



cmake_minimum_required(VERSION 3.8)
set(TARGET_NAME weightonlykernel)
project(${TARGET_NAME})

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++17 -O3 -Xcompiler "-fPIC" 
      -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 --expt-relaxed-constexpr)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(cutlass_kernels)
include_directories(utils)
include_directories(include)

include_directories(cutlass_extensions/include)
include_directories(cutlass_extensions/include/cutlass_extensions)
include_directories(cutlass/include)
include_directories(cutlass/tools/util/include)
include_directories(./)

 

cuda_add_library(
  ${TARGET_NAME} SHARED
   fpA_intB_gemm_wrapper.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int8GroupwiseColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int8GroupwiseColumnMajorInterleavedTrue.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int8PerChannelColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int8PerChannelColumnMajorInterleavedTrue.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int8PerChannelColumnMajorTrue.cu

      weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int4GroupwiseColumnMajorInterleavedTrue.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int4PerChannelColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int4PerChannelColumnMajorInterleavedTrue.cu
      weightOnlyBatchedGemv/kernelDispatcherFp16Int4PerChannelColumnMajorTrue.cu

      weightOnlyBatchedGemv/kernelDispatcherBf16Int8GroupwiseColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherBf16Int8GroupwiseColumnMajorInterleavedTrue.cu
      weightOnlyBatchedGemv/kernelDispatcherBf16Int8PerChannelColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherBf16Int8PerChannelColumnMajorInterleavedTrue.cu
    

      weightOnlyBatchedGemv/kernelDispatcherBf16Int4GroupwiseColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherBf16Int4GroupwiseColumnMajorInterleavedTrue.cu
      weightOnlyBatchedGemv/kernelDispatcherBf16Int4PerChannelColumnMajorFalse.cu
      weightOnlyBatchedGemv/kernelDispatcherBf16Int4PerChannelColumnMajorInterleavedTrue.cu
 
      
      cutlass_kernels/fpA_intB_gemm_interface.cu
      cutlass_kernels/cutlass_heuristic.cc
      cutlass_kernels/cutlass_preprocessors.cc
      utils/logger.cc
      utils/cuda_utils.cc


      generated/gemm/cutlass_kernel_file_1.generated.cu
      generated/gemm/cutlass_kernel_file_2.generated.cu

     
)


