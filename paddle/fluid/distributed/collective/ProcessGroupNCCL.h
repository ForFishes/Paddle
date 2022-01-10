//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/NCCLUtils.h"
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/event.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

namespace paddle {
namespace framework {
class LoDTensor;
class ProgramDesc;
class Scope;
class Tensor;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

// constexpr auto kNoTimeout = std::chrono::milliseconds(0);
// constexpr auto kProcessGroupDefaultTimeout =
//     std::chrono::milliseconds(30 * 60 * 1000);

constexpr const char* NCCL_BACKEND_NAME = "NCCL";
#define NCCL_UNIQUE_ID_BYTES 128

namespace paddle {
namespace distributed {
using Tensor = paddle::framework::Tensor;
using Place = paddle::platform::Place;
using CUDAStream = platform::stream::CUDAStream;
using CudaEvent = paddle::platform::CudaEvent;

class ProcessGroupNCCL : public ProcessGroup {
 public:
  class WorkNCCL : public ProcessGroup::Work,
                   public std::enable_shared_from_this<WorkNCCL> {
   public:
    WorkNCCL(const std::vector<Place>& places, int rank, OpType OpType,
             const std::vector<Tensor>& inputs);

    // bool IsStarted();
    bool IsCompleted();

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void SetOutputs(std::vector<Tensor>& outputs);  // NOLINT

    virtual ~WorkNCCL();

    std::shared_ptr<std::vector<CudaEvent>> ncclEndEvents_;

   protected:
    std::vector<Place> places_;

    std::vector<std::shared_ptr<NCCLComm>> ncclComms_;
    std::shared_ptr<std::vector<Tensor>> outputs_;

   private:
  };

  ProcessGroupNCCL(const ProcessGroupStrategy& strategy, int rank, int size);

  const std::string getBackendName() const override {
    return std::string(NCCL_BACKEND_NAME);
  }

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<Tensor>& tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

 protected:
  virtual std::shared_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      std::vector<Place> places, int rank, OpType opType,
      const std::vector<Tensor>& inputs);

 protected:
  ProcessGroupStrategy strategy_;
  std::shared_ptr<NCCLComm> nccl_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>
      places_to_ncclcomm_;

  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>
      ncclid_to_comm_;

  std::unordered_map<std::string, std::vector<std::unique_ptr<CUDAStream>>>
      places_to_streams_;

  std::unordered_map<std::string, std::vector<CudaEvent>> places_to_events_;

 private:
  void BcastNCCLId(std::vector<ncclUniqueId>& nccl_ids, int root,  // NOLINT
                   int server_fd);

  void BroadcastUniqueNCCLID(std::vector<ncclUniqueId>& nccl_ids,  // NOLINT
                             OpType opType, const std::string& p2pKey,
                             int p2pRank);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Work> collective(
      std::vector<Tensor>& inputs,   // NOLINT
      std::vector<Tensor>& outputs,  // NOLINT
      Fn fn, OpType op_type);

  std::vector<std::shared_ptr<NCCLComm>>& GetNCCLComm(
      const std::string& places_key, const std::vector<Place>& places,
      OpType opType, int p2pRank = 0, bool isSendRecvSelf = false);
};

}  //  namespace distributed
}  //  namespace paddle
