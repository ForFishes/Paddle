// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace paddle {
namespace operators {

class SystemCUDAAllocator : public phi::Allocator {
 public:
  static phi::Allocator *Instance() {
    static SystemCUDAAllocator allocator;
    return &allocator;
  }

  phi::Allocator::AllocationPtr Allocate(size_t size) override {
    if (size == 0) {
      return nullptr;
    }
    void *ptr = nullptr;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
    return phi::Allocator::AllocationPtr(new phi::Allocation(ptr, size, place),
                                         DeleteFunc);
  }

  bool IsAllocThreadSafe() const override { return true; }

 private:
  static void DeleteFunc(phi::Allocation *allocation) {
    cudaFree(allocation->ptr());
    delete allocation;
  }

  SystemCUDAAllocator() : place(platform::GetCurrentDeviceId()) {}

  DISABLE_COPY_AND_ASSIGN(SystemCUDAAllocator);

 private:
  phi::GPUPlace place;
};

template <typename T>
static __global__ void FillBarrierValue(T *x, T value) {
  x[threadIdx.x] = value;
}

template <typename T, int N>
static __forceinline__ __device__ void BarrierAllGPUs(
    const phi::Array<volatile T *, N> &barriers, T barrier_value, int rank) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  if (thread_id < N) {
    if (block_id == 0) {
      barriers[thread_id][rank] = barrier_value;
    }
    while (barriers[rank][thread_id] < barrier_value) {
    }
  }

  __syncthreads();
}

template <typename T, int N>
struct AlignedVectorAddHelper {
  DEVICE static void Run(const phi::AlignedVector<T, N> &in,
                         phi::AlignedVector<T, N> *out) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      (*out)[i] += in[i];
    }
  }
};

template <>
struct AlignedVectorAddHelper<phi::dtype::float16, 8> {
  DEVICE static void Run(const phi::AlignedVector<phi::dtype::float16, 8> &in,
                         phi::AlignedVector<phi::dtype::float16, 8> *out) {
    const __half2 *in_ptr =
        static_cast<const __half2 *>(static_cast<const void *>(&in[0]));
    __half2 *out_ptr = static_cast<__half2 *>(static_cast<void *>(&(*out)[0]));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out_ptr[i] = __hadd2(out_ptr[i], in_ptr[i]);
    }
  }
};

template <typename T, typename BarrierT, int N, int VecSize>
static __global__ void OneShotAllReduceKernel(
    phi::Array<const T *, N> ins,
    phi::Array<volatile BarrierT *, N> barriers,
    BarrierT barrier_value,
    int rank,
    size_t n,
    T *out) {
  BarrierAllGPUs<BarrierT, N>(barriers, barrier_value, rank);

  size_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  size_t stride = (blockDim.x * gridDim.x) * VecSize;
  size_t limit = n - VecSize;

  using AlignedVec = phi::AlignedVector<T, VecSize>;
  while (idx + VecSize <= n) {
    AlignedVec in_vecs[N];

#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto cur_rank = (i + rank) % N;
      const auto *ptr = ins[cur_rank] + idx;
      phi::Load(ptr, &in_vecs[cur_rank]);
    }

#pragma unroll
    for (int i = 1; i < N; ++i) {
      AlignedVectorAddHelper<T, VecSize>::Run(in_vecs[i], &in_vecs[0]);
    }
    phi::Store(in_vecs[0], out + idx);
    idx += stride;
  }

  while (idx < n) {
    T sum = ins[0][idx];
#pragma unroll
    for (int i = 1; i < N; ++i) {
      sum += ins[i][idx];
    }
    out[idx] = sum;
    ++idx;
  }
}

class CustomNCCLComm {
 public:
  virtual void SwapInput(phi::DenseTensor *x) = 0;
  virtual phi::DenseTensor AllReduce() = 0;

  virtual ~CustomNCCLComm() = default;

 protected:
  void EnableP2P(int nranks) {
    for (int i = 0; i < nranks; ++i) {
      platform::CUDADeviceGuard guard(i);
      for (int j = 0; j < nranks; ++j) {
        int enabled = 0;
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceCanAccessPeer(&enabled, i, j));
        PADDLE_ENFORCE_EQ(enabled, 1);
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
};

template <int N>
class CustomNCCLCommImpl : public CustomNCCLComm {
 private:
  template <typename T>
  struct P2PBuffer {
    template <typename InitFunc>
    P2PBuffer(CustomNCCLCommImpl<N> *comm, size_t size, InitFunc &&init_func) {
      phi::Dim<1> dim;
      dim[0] = static_cast<int64_t>(size);
      t_.Resize(dim);
      // void *ptr = comm->ctx_->template Alloc<T>(&t_);
      void *ptr =
          t_.AllocateFrom(SystemCUDAAllocator::Instance(),
                          paddle::experimental::CppTypeToDataType<T>::Type());
      init_func(*(comm->ctx_), &t_);
      comm->ctx_->Wait();

      comm->Barrier();

      auto pids = comm->AllGatherPOD(::getpid());
      for (int i = 0; i < N; ++i) {
        BroadcastDevicePtr(comm, ptr, i, pids[0]);
      }
    }

    ~P2PBuffer() {
      for (int i = 0; i < N; ++i) {
        if (i != rank_) {
          cudaIpcCloseMemHandle(ptrs_[i]);
        }
        ::munmap(mmap_ptrs_[i], sizeof(cudaIpcMemHandle_t));
        ::shm_unlink(shm_names_[i].c_str());
      }
      t_.clear();
    }

    const phi::DenseTensor &GetTensor() const { return t_; }
    phi::DenseTensor *GetMutableTensor() { return &t_; }

    template <typename U = T>
    phi::Array<U *, N> GetPtrs() const {
      phi::Array<U *, N> results;
#pragma unroll
      for (int i = 0; i < N; ++i) {
        results[i] = static_cast<U *>(ptrs_[i]);
      }
      return results;
    }

   private:
    void BroadcastDevicePtr(CustomNCCLCommImpl<N> *comm,
                            void *ptr,
                            int cur_rank,
                            pid_t pid) {
      VLOG(10) << "BroadcastDevicePtr starts " << cur_rank << " -> "
               << comm->rank_;
      std::string name = "/paddle_custom_nccl_" + std::to_string(pid) + "_" +
                         std::to_string(cur_rank);
      cudaIpcMemHandle_t *handle;
      bool is_root = (comm->rank_ == cur_rank);

      if (!is_root) {
        comm->Barrier();
      }

      int fd = ::shm_open(
          name.c_str(), is_root ? (O_RDWR | O_CREAT) : O_RDONLY, 0600);
      PADDLE_ENFORCE_GE(fd, 0);
      if (is_root) {
        PADDLE_ENFORCE_EQ(ftruncate(fd, sizeof(cudaIpcMemHandle_t)), 0);
      }
      void *mmap_ptr = ::mmap(nullptr,
                              sizeof(cudaIpcMemHandle_t),
                              is_root ? (PROT_READ | PROT_WRITE) : PROT_READ,
                              MAP_SHARED,
                              fd,
                              0);
      PADDLE_ENFORCE_NOT_NULL(mmap_ptr);
      PADDLE_ENFORCE_NE(mmap_ptr, MAP_FAILED);
      handle = static_cast<cudaIpcMemHandle_t *>(mmap_ptr);
      if (is_root) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(handle, ptr));
        ptrs_[cur_rank] = ptr;
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(
            &ptrs_[cur_rank], *handle, cudaIpcMemLazyEnablePeerAccess));
      }
      if (is_root) {
        comm->Barrier();
      }

      comm->Barrier();
      mmap_ptrs_[cur_rank] = mmap_ptr;
      shm_names_[cur_rank] = name;
      VLOG(10) << "BroadcastDevicePtr ends " << cur_rank << " -> "
               << comm->rank_;
    }

   private:
    phi::Array<void *, N> ptrs_;
    phi::DenseTensor t_;
    int rank_;
    phi::Array<void *, N> mmap_ptrs_;
    phi::Array<std::string, N> shm_names_;
  };

 public:
  using BarrierDType = uint32_t;
  using BarrierTensorDType = int32_t;

  static_assert(sizeof(BarrierDType) == sizeof(BarrierTensorDType),
                "Size not match");

  CustomNCCLCommImpl(const phi::GPUContext &ctx, size_t max_size, int ring_id)
      : ctx_(&ctx), max_size_(max_size) {
    auto comm =
        platform::NCCLCommContext::Instance().Get(ring_id, ctx.GetPlace());
    comm_ = comm->comm();
    rank_ = comm->rank();
    auto nranks = comm->nranks();
    PADDLE_ENFORCE_EQ(
        nranks,
        N,
        phi::errors::InvalidArgument("Invalid world size, this may be a bug."));

    barrier_value_ = 0;
    VLOG(10) << "CustomNCCLCommImpl::CustomNCCLCommImpl";
    ins_ = std::make_unique<P2PBuffer<uint8_t>>(
        this, max_size, [](const phi::GPUContext &ctx, phi::DenseTensor *t) {});
    VLOG(10) << "CustomNCCLCommImpl::ins_ inited";

    barriers_ = std::make_unique<P2PBuffer<BarrierTensorDType>>(
        this, N, [](const phi::GPUContext &ctx, phi::DenseTensor *t) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaMemsetAsync(t->data(),
                              0,
                              t->numel() * sizeof(BarrierTensorDType),
                              ctx.stream()));
        });
    VLOG(10) << "CustomNCCLCommImpl::barriers_ inited";
  }

  void SwapInput(phi::DenseTensor *x) override {
    out_ = *x;
    auto mem_size = x->numel() * phi::SizeOf(x->dtype());
    if (mem_size <= max_size_ && !HasReachMaxBarrierValue()) {
      ShareTensor(x, ins_->GetMutableTensor());
    }
  }

  phi::DenseTensor AllReduce() override {
    auto dtype = out_.dtype();
    auto numel = out_.numel();
    auto mem_size = numel * phi::SizeOf(dtype);
    if (mem_size > max_size_ || HasReachMaxBarrierValue()) {
      auto out_ptr = out_.data();
      auto nccl_dtype = platform::ToNCCLDataType(dtype);
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
          out_ptr, out_ptr, numel, nccl_dtype, ncclSum, comm_, ctx_->stream()));
      VLOG(10) << "Use ncclAllReduce since " << mem_size << " > " << max_size_;
      if (HasReachMaxBarrierValue()) {
        Barrier();
        auto *barrier_tensor = barriers_->GetMutableTensor();
        PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
            barrier_tensor->data(),
            0,
            barrier_tensor->numel() * sizeof(BarrierTensorDType),
            ctx_->stream()));
        Barrier();
        barrier_value_ = 0;
      }
      return std::move(out_);
    }

#define PD_CUSTOM_ALLREDUCE(__cpp_dtype, __vec_size)                      \
  do {                                                                    \
    if (dtype ==                                                          \
        ::paddle::experimental::CppTypeToDataType<__cpp_dtype>::Type()) { \
      return AllReduceImpl<__cpp_dtype, __vec_size>();                    \
    }                                                                     \
  } while (0)

    VLOG(10) << "Use custom AllReduce since " << mem_size
             << " <= " << max_size_;
    PD_CUSTOM_ALLREDUCE(phi::dtype::float16, 8);
    PD_CUSTOM_ALLREDUCE(float, 4);
    PD_CUSTOM_ALLREDUCE(double, 2);
    PADDLE_THROW(
        phi::errors::InvalidArgument("Unsupported data type %s", dtype));
  }

 private:
  bool HasReachMaxBarrierValue() const {
    return barrier_value_ == std::numeric_limits<BarrierDType>::max();
  }

  template <typename T, int VecSize>
  phi::DenseTensor AllReduceImpl() {
    const auto &in_ptrs = ins_->template GetPtrs<const T>();
    const auto &barrier_ptrs =
        barriers_->template GetPtrs<volatile BarrierDType>();
    auto *out_data = out_.template data<T>();
    ++barrier_value_;

    int64_t numel = out_.numel();
    int threads = ctx_->GetMaxThreadsPerBlock();
    PADDLE_ENFORCE_GE(threads, N);
    int64_t blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;
    blocks = std::min<int64_t>(blocks, ctx_->GetCUDAMaxGridDimSize()[0]);
    OneShotAllReduceKernel<T, BarrierDType, N, VecSize>
        <<<blocks, threads, 0, ctx_->stream()>>>(in_ptrs,
                                                 barrier_ptrs,
                                                 barrier_value_,
                                                 rank_,
                                                 out_.numel(),
                                                 out_data);
    return std::move(out_);
  }

  void ShareTensor(phi::DenseTensor *x, phi::DenseTensor *y) {
    PADDLE_ENFORCE_LE(x->numel(), max_size_);
    const void *y_ptr = y->data();
    PADDLE_ENFORCE_LE(x->numel() * phi::SizeOf(x->dtype()),
                      y->numel() * phi::SizeOf(y->dtype()));
    y->Resize(x->dims());
    auto *new_y_ptr = ctx_->Alloc(y, x->dtype());
    PADDLE_ENFORCE_EQ(y_ptr, new_y_ptr);
    x->ShareBufferWith(*y);
  }

  void Barrier() { AllGatherPOD(1); }

  template <typename T>
  std::vector<T> AllGatherPOD(const T &value) {
    std::vector<T> result(N);
    AllGatherBuffer(&value, result.data(), sizeof(T));
    return result;
  }

  void AllGatherBuffer(const void *src, void *dst, size_t nbytes) {
    phi::DenseTensor tensor;
    phi::Dim<1> dim;
    dim[0] = N * nbytes;
    tensor.Resize(dim);
    auto *ptr = ctx_->template Alloc<uint8_t>(&tensor);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(ptr + rank_ * nbytes,
                                               src,
                                               nbytes,
                                               cudaMemcpyHostToDevice,
                                               ctx_->stream()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllGather(
        ptr + rank_ * nbytes, ptr, nbytes, ncclInt8, comm_, ctx_->stream()));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        dst, ptr, N * nbytes, cudaMemcpyDeviceToHost, ctx_->stream()));
    ctx_->Wait();
  }

 private:
  std::unique_ptr<P2PBuffer<uint8_t>> ins_;
  std::unique_ptr<P2PBuffer<BarrierTensorDType>> barriers_;
  BarrierDType barrier_value_;
  phi::DenseTensor out_;

  const phi::GPUContext *ctx_;
  size_t max_size_;
  ncclComm_t comm_;
  int rank_;
};

static std::unique_ptr<CustomNCCLComm> CreateCustomNCCLComm(
    const phi::GPUContext &ctx, int64_t max_size, int ring_id) {
  if (max_size <= 0) {
    return nullptr;
  }

  auto nranks = platform::NCCLCommContext::Instance()
                    .Get(ring_id, ctx.GetPlace())
                    ->nranks();
#define PD_CREATE_CUSTOM_NCCL_COMM(__nranks)                 \
  do {                                                       \
    if (nranks == __nranks) {                                \
      return std::make_unique<CustomNCCLCommImpl<__nranks>>( \
          ctx, max_size, ring_id);                           \
    }                                                        \
  } while (0)

  PD_CREATE_CUSTOM_NCCL_COMM(8);
  PD_CREATE_CUSTOM_NCCL_COMM(4);
  PD_CREATE_CUSTOM_NCCL_COMM(2);
  return nullptr;
}

}  // namespace operators
}  // namespace paddle
