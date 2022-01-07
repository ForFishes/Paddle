/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/pybind/distributed_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindDistributed(py::module *m) {
  py::enum_<distributed::ReduceOp>(*m, "ReduceOp")
      .value("SUM", distributed::ReduceOp::SUM)
      .value("AVG", distributed::ReduceOp::AVG)
      .value("MAX", distributed::ReduceOp::MAX)
      .value("MIN", distributed::ReduceOp::MIN);

  py::class_<distributed::AllreduceOptions>(*m, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &distributed::AllreduceOptions::reduceOp)
      .def_readwrite("timeout", &distributed::AllreduceOptions::timeout);

  auto processGroup =
      py::class_<distributed::ProcessGroup,
                 std::shared_ptr<distributed::ProcessGroup>>(*m, "ProcessGroup")
          // .def(py::init<int, int>())
          .def("rank", &distributed::ProcessGroup::getRank)
          .def("size", &distributed::ProcessGroup::getSize)
          .def("name", &distributed::ProcessGroup::getBackendName)
          .def(
              "allreduce",
              [](distributed::ProcessGroup &self, imperative::VarBase &vb,
                 distributed::ReduceOp op) {
                distributed::AllreduceOptions opts;
                opts.reduceOp = op;
                auto *x =
                    vb.MutableVar()->GetMutable<paddle::framework::LoDTensor>();
                std::vector<paddle::framework::Tensor> ts = {*x};
                return self.allreduce(ts, opts);
              },
              py::arg("vb"), py::arg("op") = distributed::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>());

  auto processGroupNCCL =
      py::class_<distributed::ProcessGroupNCCL,
                 std::shared_ptr<distributed::ProcessGroupNCCL>>(
          *m, "ProcessGroupNCCL", processGroup)
          .def(py::init<const distributed::ProcessGroupStrategy &, int, int>(),
               py::call_guard<py::gil_scoped_release>());

  // define parallel strategy, it will be removed
  py::class_<distributed::ProcessGroupStrategy> pg_strategy(
      *m, "ProcessGroupStrategy", "");
  pg_strategy.def(py::init())
      .def_property("nranks",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.nranks_;
                    },
                    [](distributed::ProcessGroupStrategy &self, int nranks) {
                      self.nranks_ = nranks;
                    })
      .def_property("local_rank",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.local_rank_;
                    },
                    [](distributed::ProcessGroupStrategy &self,
                       int local_rank) { self.local_rank_ = local_rank; })
      .def_property(
          "trainer_endpoints",
          [](const distributed::ProcessGroupStrategy &self) {
            return self.trainer_endpoints_;
          },
          [](distributed::ProcessGroupStrategy &self,
             std::vector<std::string> eps) { self.trainer_endpoints_ = eps; })
      .def_property("current_endpoint",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.current_endpoint_;
                    },
                    [](distributed::ProcessGroupStrategy &self,
                       const std::string &ep) { self.current_endpoint_ = ep; })
      .def_property("nrings",
                    [](const distributed::ProcessGroupStrategy &self) {
                      return self.nrings_;
                    },
                    [](distributed::ProcessGroupStrategy &self, int nrings) {
                      self.nrings_ = nrings;
                    });
}

}  // end namespace pybind
}  // namespace paddle
