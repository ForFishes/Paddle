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

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/imperative/distributed/ProcessGroupNCCL.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

std::string buildNcclUniqueIdStr(const ncclUniqueId &ncclID) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&ncclID);
  std::ostringstream oss;
  // for(const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
  //   oss << std::hex << static_cast<int>(bytes[i]);
  // }
  for (auto i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

ProcessGroupNCCL::ProcessGroupNCCL(const ProcessGroupStrategy &strategy,
                                   int rank, int size)
    : ProcessGroup(rank, size), strategy_(strategy) {
  // construct ncllcomm
  Init();
}

void ProcessGroupNCCL::BcastNCCLId(
    std::vector<ncclUniqueId> &nccl_ids,  // NOLINT
    int root, int server_fd) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &nccl_ids);
  } else {
    platform::RecvBroadCastCommID(server_fd, strategy_.current_endpoint_,
                                  &nccl_ids);
  }
}

void ProcessGroupNCCL::Init() {
  int server_fd = -1;

  std::vector<ncclUniqueId> nccl_ids;
  nccl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique ncclid on the root worker
    for (size_t i = 0; i < nccl_ids.size(); ++i) {
      platform::dynload::ncclGetUniqueId(&nccl_ids[i]);
    }
  } else {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastNCCLId(nccl_ids, 0, server_fd);

  VLOG(1) << "init nccl rank:" << strategy_.local_rank_
          << ", nranks:" << strategy_.nranks_
          << ", nccl uniqueid: " << buildNcclUniqueIdStr(nccl_ids.front());

  nccl_comm_ = NCCLComm::create(strategy_.nranks_, strategy_.local_rank_,
                                nccl_ids.front());
}

}  //  namespace imperative
}  //  namespace paddle
