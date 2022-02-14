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

#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"

DECLARE_bool(nccl_blocking_wait);
constexpr int64_t kSynchronizeBusyWaitMillis = 10;

namespace paddle {
namespace distributed {

// NCCL op mapping
const std::map<ReduceOp, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
};

ncclDataType_t ToNCCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return ncclFloat;
  } else if (type == framework::proto::VarType::FP64) {
    return ncclDouble;
  } else if (type == framework::proto::VarType::INT32) {
    return ncclInt;
  } else if (type == framework::proto::VarType::INT64) {
    return ncclInt64;
  } else if (type == framework::proto::VarType::FP16) {
    return ncclFloat16;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in nccl is not supported."));
  }
}

std::string BuildNcclUniqueIdStr(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (auto i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

// Get the list of devices from list of tensors
std::vector<Place> GetPlaceList(const std::vector<Tensor>& tensors) {
  std::vector<Place> places;
  places.reserve(tensors.size());
  for (auto& tensor : tensors) {
    places.push_back(tensor.place());
  }
  return places;
}

// Get the deviceList String from the list of devices
std::string GetKeyFromPlaces(const std::vector<Place>& places) {
  std::string placeList;
  for (auto& place : places) {
    std::stringstream tmp;
    tmp << place;
    if (placeList.empty()) {
      placeList += tmp.str();
    } else {
      placeList += "," + tmp.str();
    }
  }
  return placeList;
}

void SyncStreams(
    const std::vector<Place>& places,
    std::vector<CudaEvent>& ncclEvents,                       // NOLINT
    std::vector<std::unique_ptr<CUDAStream>>& ncclStreams) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    auto cuda_ctx = dev_ctx->context();
    auto& curr_stream = cuda_ctx->Stream();
    ncclEvents[i].Record(curr_stream);
    ncclEvents[i].Block(ncclStreams[i]);
  }
}

std::shared_ptr<ProcessGroupNCCL::WorkNCCL> ProcessGroupNCCL::CreateWork(
    std::vector<Place> places, int rank, OpType opType,
    const std::vector<Tensor>& inputs) {
  return std::make_shared<ProcessGroupNCCL::WorkNCCL>(places, rank, opType,
                                                      inputs);
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const std::vector<Place>& places, int rank,
                                     OpType OpType,
                                     const std::vector<Tensor>& inputs)
    : Work(rank, inputs, OpType),
      places_(places),
      start_time_(std::chrono::steady_clock::now()) {
  ncclEndEvents_ = std::make_shared<std::vector<CudaEvent>>(places.size());
  ncclComms_.resize(places.size());
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() {}

void ProcessGroupNCCL::WorkNCCL::SetOutputs(
    std::vector<Tensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<Tensor>>(outputs);
}

void ProcessGroupNCCL::WorkNCCL::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    auto cuda_ctx = dev_ctx->context();
    auto& curr_stream = cuda_ctx->Stream();
    (*ncclEndEvents_)[i].Block(curr_stream);
  }
}

bool ProcessGroupNCCL::WorkNCCL::IsCompleted() {
  for (size_t i = 0; i < places_.size(); ++i) {
    if (!(*ncclEndEvents_)[i].Query()) {
      return false;
    }
  }

  return true;
}

bool ProcessGroupNCCL::WorkNCCL::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();
  if (FLAGS_nccl_blocking_wait) {
    while (!IsCompleted()) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  }
  return true;
}

// Same as Wait
void ProcessGroupNCCL::WorkNCCL::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupNCCL::ProcessGroupNCCL(const ProcessGroupStrategy& strategy,
                                   int rank, int size)
    : ProcessGroup(rank, size), strategy_(strategy) {}

void ProcessGroupNCCL::BcastNCCLId(
    std::vector<ncclUniqueId>& nccl_ids,  // NOLINT
    int root, int server_fd) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto& ep : strategy_.trainer_endpoints_) {
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

void ProcessGroupNCCL::BroadcastUniqueNCCLID(
    std::vector<ncclUniqueId>& nccl_ids,  // NOLINT
    OpType opType) {
  int server_fd = -1;
  if (rank_ != 0) {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastNCCLId(nccl_ids, 0, server_fd);
}

std::vector<std::shared_ptr<NCCLComm>>& ProcessGroupNCCL::GetNCCLComm(
    const std::string& places_key, const std::vector<Place>& places,
    OpType opType) {
  PADDLE_ENFORCE_EQ(places_key.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the NCCL Communicator since "
                        "the GPU place are not known"));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_ncclcomm_.find(places_key) != places_to_ncclcomm_.end()) {
      return places_to_ncclcomm_[places_key];
    }
  }

  // NCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<NCCLComm>> ncclComms;
  ncclComms.resize(places.size());

  // using vector just for broadcast
  std::vector<ncclUniqueId> nccl_ids;
  nccl_ids.resize(1);
  auto& nccl_id = nccl_ids.front();

  if (rank_ == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGetUniqueId(&nccl_id));
  }
  BroadcastUniqueNCCLID(nccl_ids, opType);

  VLOG(3) << "init nccl rank: " << strategy_.local_rank_
          << ", nranks: " << strategy_.nranks_ << ", place: " << places_key
          << ", nccl uniqueid: " << BuildNcclUniqueIdStr(nccl_id);

  std::vector<std::unique_ptr<CUDAStream>> streams;
  streams.resize(places.size());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());

  for (size_t i = 0; i < places.size(); ++i) {
    auto dev_id = places[i].device;
    platform::CUDADeviceGuard guard(dev_id);
    ncclComms[i] = NCCLComm::create(getSize(), getRank(), nccl_id);
    streams[i].reset(new CUDAStream(places[i]));
  }

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

  // create events for sync
  std::vector<CudaEvent> events;
  events.resize(places.size());

  places_to_streams_.emplace(places_key, std::move(streams));
  places_to_events_.emplace(places_key, std::move(events));
  places_to_ncclcomm_.emplace(places_key, std::move(ncclComms));
  return places_to_ncclcomm_[places_key];
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::collective(
    std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Fn fn,
    OpType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);
  auto& nccl_comms = GetNCCLComm(key, places, op_type);
  SyncStreams(places, places_to_events_[key], places_to_streams_[key]);

  auto work = CreateWork(places, rank_, op_type, inputs);
  work->SetOutputs(outputs);

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& nccl_stream = places_to_streams_[key][i];
    fn(inputs[i], outputs[i], nccl_comms[i]->getNcclComm(), nccl_stream);
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& nccl_stream = places_to_streams_[key][i];
    (*work->ncclEndEvents_)[i].Record(nccl_stream);
  }

  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce(
    std::vector<Tensor>& tensors, const AllreduceOptions& opts) {
  auto tensor = tensors.back();
  auto place = tensor.place();
  return collective(tensors, tensors,
                    [&](Tensor& input, Tensor& output, ncclComm_t comm,
                        std::unique_ptr<CUDAStream>& stream) {
                      return platform::dynload::ncclAllReduce(
                          input.mutable_data(place), output.mutable_data(place),
                          input.numel(), ToNCCLDataType(input.type()),
                          ncclOp.at(opts.reduceOp), comm, stream->raw_stream());
                    },
                    OpType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::broadcast(
    std::vector<Tensor>& tensors, const BroadcastOptions& opts) {
  auto tensor = tensors.back();
  auto place = tensor.place();
  return collective(tensors, tensors,
                    [&](Tensor& input, Tensor& output, ncclComm_t comm,
                        std::unique_ptr<CUDAStream>& stream) {
                      const auto root = opts.source_rank * tensors.size();
                      return platform::dynload::ncclBcast(
                          input.mutable_data(place), input.numel(),
                          ToNCCLDataType(input.type()), root, comm,
                          stream->raw_stream());
                    },
                    OpType::BROADCAST);
}

}  //  namespace distributed
}  //  namespace paddle
