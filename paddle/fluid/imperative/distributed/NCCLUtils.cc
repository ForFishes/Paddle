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

#include "paddle/fluid/imperative/distributed/NCCLUtils.h"

namespace paddle {
namespace imperative {

std::string getNcclVersion() {
  static std::once_flag ncclGetVersionFlag;
  static std::string versionString;

  std::call_once(ncclGetVersionFlag, []() {
    int version;
    ncclResult_t status = platform::dynload::ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      auto ncclMajor = version / 1000;
      auto ncclMinor = (version % 1000) / 100;
      auto ncclPatch = version % (ncclMajor * 1000 + ncclMinor * 100);
      versionString = std::to_string(ncclMajor) + "." +
                      std::to_string(ncclMinor) + "." +
                      std::to_string(ncclPatch);
    }
  });

  return versionString;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(platform::dynload::ncclGetErrorString(error)) +
         ", NCCL version " + getNcclVersion();
}

ncclComm_t NCCLComm::getNcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    auto commFailureMsg =
        commFailureReason_ != ""
            ? std::string(" Original reason for failure was: ") +
                  commFailureReason_
            : "";
  }
  return ncclComm_;
}

}  // namespace imperative
}  // namespace paddle
