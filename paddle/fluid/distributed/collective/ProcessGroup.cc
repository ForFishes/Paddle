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

#include "paddle/fluid/distributed/collective/ProcessGroup.h"

namespace paddle {
namespace distributed {

ProcessGroup::Work::Work(int rank,
                         const std::vector<framework::Tensor>& inputTensors,
                         OpType opType)
    : rank_(rank), opType_(opType) {}

ProcessGroup::Work::~Work() = default;

bool ProcessGroup::Work::IsCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return
}

ProcessGroup::ProcessGroup(int rank, int size) : rank_(rank), size_(size) {}

}  //  namespace distributed
}  //  namespace paddle
