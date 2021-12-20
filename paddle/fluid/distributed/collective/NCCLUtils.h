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

#include <error.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/variant.hpp"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

std::string getNcclVersion();
std::string ncclGetErrorWithVersion(ncclResult_t error);

class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm)
      : ncclComm_(ncclComm),
        aborted_(false),
        ncclAsyncErr_(ncclSuccess),
        commFailureReason_("") {}

  NCCLComm() : NCCLComm(nullptr) {}

  ~NCCLComm() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ncclComm_ && !aborted_) {
      platform::dynload::ncclCommDestroy(ncclComm_);
    }
  }

  static std::shared_ptr<NCCLComm> create(int numRanks, int rank,
                                          ncclUniqueId commId) {
    auto comm = std::make_shared<NCCLComm>();

    ncclResult_t result = platform::dynload::ncclCommInitRank(
        &(comm->ncclComm_), numRanks, commId, rank);

    PADDLE_ENFORCE_EQ(
        result, ncclSuccess,
        platform::errors::Fatal("NCCL error in: " + std::string(__FILE__) +
                                ":" + std::to_string(__LINE__) + ", " +
                                ncclGetErrorWithVersion(result) + "\n"));

    comm->ncclId_ = commId;
    comm->rank_ = rank;
    return comm;
  }

  ncclUniqueId getNcclId() { return ncclId_; }

  // Must not be copyable
  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  NCCLComm& operator=(NCCLComm&& other) = delete;

  // Move constructable
  NCCLComm(NCCLComm&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(ncclComm_, other.ncclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
  }

  ncclComm_t getNcclComm();

  std::string getNcclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return commFailureReason_;
  }

  void ncclCommAbort(std::string commFailureReason = "") {
    std::unique_lock<std::mutex> lock(mutex_);
    return;
  }

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  ncclResult_t checkForNcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
    return ncclSuccess;
  }

 protected:
  ncclComm_t ncclComm_;
  ncclUniqueId ncclId_;
  bool aborted_;
  ncclResult_t ncclAsyncErr_;
  int rank_;
  mutable std::mutex mutex_;
  std::string commFailureReason_;
};

}  // namespace distributed
}  // namespace paddle
