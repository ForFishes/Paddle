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

#include <map>
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace framework {

static size_t GetTensorMemSize(const phi::DenseTensor &t) {
  if (!t.IsInitialized() || t.place() == phi::CPUPlace()) {
    return 0;
  } else {
    return t.numel() * phi::SizeOf(t.dtype());
  }
}

static void CollectScopeLocalVarMemSize(const Scope &scope,
                                        bool include_kids,
                                        std::map<std::string, size_t> *ret) {
  const auto &names = scope.LocalVarNames();
  for (const auto &name : names) {
    const auto *var = scope.FindLocalVar(name);
    if (var->IsType<phi::DenseTensor>()) {
      (*ret)["tensor"] += GetTensorMemSize(var->Get<phi::DenseTensor>());
    } else if (var->IsType<LoDTensorArray>()) {
      const auto &array = var->Get<LoDTensorArray>();
      auto &array_mem = (*ret)["tensor_array"];
      for (const auto &t : array) {
        array_mem += GetTensorMemSize(t);
      }
    }
  }
  if (include_kids) {
    for (const auto &kid : scope.kids()) {
      CollectScopeLocalVarMemSize(*kid, true, ret);
    }
  }
}

static std::string MemInfoMapToJSONString(
    const std::map<std::string, size_t> &mem_info) {
  std::string json_ret;
  size_t i = 0;
  for (const auto &pair : mem_info) {
    json_ret += "\"" + pair.first + "\": " + std::to_string(pair.second);
    if (i + 1 != mem_info.size()) {
      json_ret += ", ";
    }
    ++i;
  }
  return "{" + json_ret + "}";
}

static auto GetScopeLocalVarMemSize(const Scope &scope,
                                    bool include_kids = false) {
  std::map<std::string, size_t> ret({{"tensor", 0}, {"tensor_array", 0}});
  CollectScopeLocalVarMemSize(scope, include_kids, &ret);
  return ret;
}

static std::string GetAllScopeMemSize(const Scope &scope) {
  const Scope *parent = &scope;
  while (parent->parent() != nullptr) {
    parent = parent->parent();
  }
  auto ret = GetScopeLocalVarMemSize(*parent, true);
  return MemInfoMapToJSONString(ret);
}

}  // namespace framework
}  // namespace paddle
