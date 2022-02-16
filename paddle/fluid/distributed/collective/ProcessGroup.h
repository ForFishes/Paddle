//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"

constexpr auto kWaitTimeout = std::chrono::milliseconds(0);
constexpr auto kProcessGroupDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace paddle {
namespace distributed {

enum class OpType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_SPARSE = 2,
  REDUCE = 3,
  ALLGATHER = 4,
  GATHER = 5,
  SCATTER = 6,
  REDUCE_SCATTER = 7,
  ALLTOALL = 8,
  SEND = 9,
  RECV = 10,
  BARRIER = 11,
  UNKNOWN = 100,
};

struct ProcessGroupStrategy {
  int nranks_{1};
  int local_rank_{0};
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoint_{""};
  int nrings_{1};
};

class ProcessGroup {
 public:
  class Task {
   public:
    Task(int rank, const std::vector<framework::Tensor>& inputTensors,
         OpType opType = OpType::UNKNOWN);

    virtual ~Task();
    virtual bool IsCompleted();
    virtual bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);
    virtual void Synchronize();

   protected:
    const int rank_;
    OpType opType_;
    std::mutex mutex_;
    bool is_completed_ = false;
  };

  explicit ProcessGroup(int rank, int size);
  virtual ~ProcessGroup() {}

  int getRank() const { return rank_; }

  int getSize() const { return size_; }

  // subclass must override this method to return the backend name
  virtual const std::string getBackendName() const = 0;

  virtual std::shared_ptr<ProcessGroup::Task> allreduce(
      std::vector<paddle::framework::Tensor>& /* tensors */,
      const AllreduceOptions& = AllreduceOptions()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support allreduce", getBackendName()));
  }

  virtual std::shared_ptr<ProcessGroup::Task> broadcast(
      std::vector<paddle::framework::Tensor>& /* tensors */,
      const BroadcastOptions& = BroadcastOptions()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "ProcessGroup%s does not support allreduce", getBackendName()));
  }

 protected:
  const int rank_;
  const int size_;
};

}  //  namespace distributed
}  //  namespace paddle
