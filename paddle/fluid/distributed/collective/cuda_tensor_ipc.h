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

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/api/utils/tensor_utils.h"  // NOTE: this header is required somewhere
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;


class CudaTensorIPC {
    public:
        CudaTensorIPC(const phi::DenseTensor& tensor){
            size_ = tensor.numel();
            type_ = tensor.type();
            tensor_ = tensor;
            VLOG(0) << "CudaTensorIPC: size: " << size_ << ", type: " << type;
            cudaIpcGetMemHandle(&ipc_handle_, const_cast<void *>(tensor.data()));
        }

        CudaTensorIPC(const cudaIpcMemHandle_t& ipc_handle, size_t size, phi::DataType type)
            : ipc_handle_(ipc_handle), size_(size), type_(type) {
            VLOG(0) << "CudaTensorIPC: size: " << size_ << ", type: " << type;
            

        }

        ~CudaTensorIPC();

        void SetTensorSize(size_t size) { size_ = size; }

        void SetDataType(phi::DataType type) { this->type_ = type; }

        size_t GetTensorSize() const { return size_; }

        phi::DataType GetDataType() const { return type_; }

        cudaIpcMemHandle_t GetIpcHandle() const { return ipc_handle_; }

  

    private:
        cudaIpcMemHandle_t ipc_handle_;
        size_t size_;
        phi::DataType type_;
        phi::DenseTensor tensor_;
};





}  //  namespace distributed
}  //  namespace paddle