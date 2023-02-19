/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/matmul_kernel.h"

#include <mutex>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas_impl.cu.h"
#endif

namespace phi {

#ifdef PADDLE_WITH_CUDA
static std::vector<cudaStream_t> g_streams;
static std::vector<cudaEvent_t> g_events;
static std::mutex g_mtx;

static void GetStreamsAndEvents(cudaStream_t **streams,
                                cudaEvent_t **events,
                                size_t n) {
  std::lock_guard<std::mutex> guard(g_mtx);
  if (g_streams.size() < n) {
    auto origin_n = g_streams.size();
    g_streams.resize(n);
    for (size_t i = origin_n; i < n; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamCreateWithFlags(&g_streams[i], cudaStreamNonBlocking));
    }
  }

  if (g_events.size() <= n) {
    auto origin_n = g_events.size();
    g_events.resize(n + 1);
    for (size_t i = origin_n; i < n + 1; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&g_events[i], cudaEventDisableTiming));
    }
  }

  *streams = g_streams.data();
  *events = g_events.data();
}

template <typename T>
struct GEMMTrait;

template <>
struct GEMMTrait<float> {
  static float One() { return 1; }
  static float Zero() { return 0; }
  static auto DataType() { return CUDA_R_32F; }
  static auto ComputeType() { return CUDA_R_32F; }
};

template <>
struct GEMMTrait<double> {
  static double One() { return 1; }
  static double Zero() { return 0; }
  static auto DataType() { return CUDA_R_64F; }
  static auto ComputeType() { return CUDA_R_64F; }
};

template <>
struct GEMMTrait<phi::dtype::float16> {
  static float One() { return 1; }
  static float Zero() { return 0; }
  static auto DataType() { return CUDA_R_16F; }
  static auto ComputeType() { return CUDA_R_32F; }
};

template <>
struct GEMMTrait<phi::dtype::bfloat16> {
  static float One() { return 1; }
  static float Zero() { return 0; }
  static auto DataType() { return CUDA_R_16BF; }
  static auto ComputeType() { return CUDA_R_32F; }
};

template <typename T>
static void GEMMWithStride(const GPUContext &ctx,
                           const T *x,
                           const T *y,
                           T *z,
                           cudaStream_t stream,
                           int m,
                           int n,
                           int k,
                           int x_stride = -1,
                           int y_stride = -1,
                           int z_stride = -1) {
  // LOG(INFO) << "[m, n, k] = [" << m << ", " << n << ", " << k << "]";
  // LOG(INFO) << "strides = [" << (x_stride < 0 ? n : x_stride) << ", "
  //           << ", " << (y_stride < 0 ? k : y_stride) << ", "
  //           << (z_stride < 0 ? k : z_stride) << "]";
  auto *gpu_ctx = const_cast<GPUContext *>(&ctx);
  const auto alpha = GEMMTrait<T>::One();
  const auto beta = GEMMTrait<T>::Zero();
  auto dtype = GEMMTrait<T>::DataType();
  auto compute_dtype = GEMMTrait<T>::ComputeType();

  auto old_stream = gpu_ctx->stream();
  if (old_stream != stream) {
    gpu_ctx->TensorCoreCublasCallIfAvailable([stream](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSetStream(handle, stream));
    });
  }

  phi::funcs::CUBlas<T>::GEMM_EX(gpu_ctx,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 k,
                                 m,
                                 n,
                                 &alpha,
                                 y,
                                 dtype,
                                 y_stride < 0 ? k : y_stride,
                                 x,
                                 dtype,
                                 x_stride < 0 ? n : x_stride,
                                 &beta,
                                 z,
                                 dtype,
                                 z_stride < 0 ? k : z_stride,
                                 compute_dtype);
  if (old_stream != stream) {
    gpu_ctx->TensorCoreCublasCallIfAvailable(
        [old_stream](cublasHandle_t handle) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::cublasSetStream(handle, old_stream));
        });
  }
}

static auto PreRecordEvent(const GPUContext &ctx, int parallel_num) {
  cudaStream_t *other_streams;
  cudaEvent_t *other_events;
  GetStreamsAndEvents(&other_streams, &other_events, parallel_num - 1);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(other_events[parallel_num - 1], ctx.stream()));
  for (int i = 0; i < parallel_num - 1; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(other_streams[i], other_events[parallel_num - 1]));
  }
  return other_streams;
}

static void PostRecordEvent(const GPUContext &ctx, int parallel_num) {
  cudaStream_t *other_streams;
  cudaEvent_t *other_events;
  GetStreamsAndEvents(&other_streams, &other_events, parallel_num - 1);
  auto main_stream = ctx.stream();
  for (int i = 0; i < parallel_num - 1; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventRecord(other_events[i], other_streams[i]));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(main_stream, other_events[i]));
  }
}

template <typename T>
void ColumnParallelMatmul(const GPUContext &ctx,
                          const T *x_,
                          const T *y_,
                          T *z_,
                          int m,
                          int n,
                          int k,
                          int parallel_num) {
  using DstT = phi::dtype::float16;

  const auto *x = reinterpret_cast<const DstT *>(static_cast<const void *>(x_));
  const auto *y = reinterpret_cast<const DstT *>(static_cast<const void *>(y_));
  auto *z = reinterpret_cast<DstT *>(static_cast<void *>(z_));

  constexpr bool kIsFP16 = std::is_same<T, phi::dtype::float16>::value;
  PADDLE_ENFORCE_EQ(kIsFP16, true);

  // parallel_num = 1;

  PADDLE_ENFORCE_EQ(k % parallel_num, 0);
  PADDLE_ENFORCE_GT(parallel_num, 1);
  auto main_stream = ctx.stream();

  // parallel_num = 1;
  int split_k = k / parallel_num;
  VLOG(5) << "PreRecordEvent starts";
  auto streams = PreRecordEvent(ctx, parallel_num);
  VLOG(5) << "PreRecordEvent ends";
  for (int i = 0; i < parallel_num; ++i) {
    const auto *cur_y = y + i * split_k;
    auto *cur_z = z + i * split_k;
    GEMMWithStride(ctx,
                   x,
                   cur_y,
                   cur_z,
                   i == 0 ? main_stream : streams[i - 1],
                   m,
                   n,
                   split_k,
                   -1,
                   k,
                   k);
  }
  PostRecordEvent(ctx, parallel_num);
}

template <typename T>
static std::vector<DenseTensor> SplitTensors(
    const GPUContext &ctx, const T *x, int m, int n, int split_num) {
  PADDLE_ENFORCE_EQ(n % split_num, 0);
  int per_n = n / split_num;
  auto stream = ctx.stream();
  std::vector<DenseTensor> tensors(split_num);
  for (int i = 0; i < split_num; ++i) {
    tensors[i].Resize({m, per_n});
    auto *dst = ctx.template Alloc<T>(&tensors[i]);
    for (int j = 0; j < m; ++j) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(dst + j * per_n,
                                                 x + j * n + i * per_n,
                                                 per_n * sizeof(T),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream));
    }
  }
  return tensors;
}

template <typename T>
void RowParallelMatmul(const GPUContext &ctx,
                       const T *x_,
                       const T *y_,
                       DenseTensor *z_,
                       int m,
                       int n,
                       int k,
                       int parallel_num) {
  PADDLE_ENFORCE_EQ(n % parallel_num, 0);
  PADDLE_ENFORCE_GT(parallel_num, 1);
  using DstT = phi::dtype::float16;

  const auto *x = reinterpret_cast<const DstT *>(static_cast<const void *>(x_));
  const auto *y = reinterpret_cast<const DstT *>(static_cast<const void *>(y_));
  auto *z = ctx.template Alloc<DstT>(z_);
  const auto &z_dims = z_->dims();
  PADDLE_ENFORCE_EQ(z_dims[0], m);
  PADDLE_ENFORCE_EQ(z_dims[1], k);

  constexpr bool kIsFP16 = std::is_same<T, phi::dtype::float16>::value;
  PADDLE_ENFORCE_EQ(kIsFP16, true);

  auto main_stream = ctx.stream();
  int split_n = n / parallel_num;

  auto split_tensors = SplitTensors<DstT>(ctx, x, m, n, parallel_num);
  std::vector<DenseTensor> other_tensors(parallel_num - 1);
  std::vector<DstT *> other_ptrs(parallel_num - 1);
  for (int i = 0; i < parallel_num - 1; ++i) {
    other_tensors[i].Resize({m, k});
    other_ptrs[i] = ctx.template Alloc<DstT>(&other_tensors[i]);
  }

  auto streams = PreRecordEvent(ctx, parallel_num);
  for (int i = 0; i < parallel_num; ++i) {
    GEMMWithStride(ctx,
                   static_cast<const DstT *>(split_tensors[i].data()),
                   y + i * split_n * k,
                   (i == 0 ? z : other_ptrs[i - 1]),
                   (i == 0 ? ctx.stream() : streams[i - 1]),
                   m,
                   split_n,
                   k);
  }
  PostRecordEvent(ctx, parallel_num);
  for (int i = 0; i < parallel_num - 1; ++i) {
    AddKernel<DstT>(ctx, *z_, other_tensors[i], z_);
  }
}

#define INIT_P_MATMUL(__type)                                    \
  template void ColumnParallelMatmul<__type>(const GPUContext &, \
                                             const __type *,     \
                                             const __type *,     \
                                             __type *,           \
                                             int,                \
                                             int,                \
                                             int,                \
                                             int);               \
  template void RowParallelMatmul<__type>(const GPUContext &,    \
                                          const __type *,        \
                                          const __type *,        \
                                          DenseTensor *,         \
                                          int,                   \
                                          int,                   \
                                          int,                   \
                                          int);

INIT_P_MATMUL(phi::dtype::bfloat16);
INIT_P_MATMUL(phi::dtype::float16);
INIT_P_MATMUL(float);
INIT_P_MATMUL(double);
INIT_P_MATMUL(phi::dtype::complex<float>);
INIT_P_MATMUL(phi::dtype::complex<double>);

#endif

}  // namespace phi

PD_REGISTER_KERNEL(matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_with_flatten,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
