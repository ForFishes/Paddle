# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import unittest
import os
import numpy as np
import random

import paddle
import paddle.nn as nn
from paddle.fluid.dygraph.nn import Linear
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard
import paddle.distributed as dist
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.optimizer import SGD
from paddle.fluid.initializer import NumpyArrayInitializer


def init_process_group(strategy=None):
    # this will remove
    if strategy is None:
        strategy = core.ProcessGroupStrategy()
        strategy.nranks = ParallelEnv().nranks
        strategy.local_rank = ParallelEnv().local_rank
        strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
        strategy.current_endpoint = ParallelEnv().current_endpoint
    if strategy.nranks < 2:
        return

    group = core.ProcessGroupNCCL(strategy, strategy.local_rank,
                                  strategy.nranks)
    return group


class LinearModel(nn.Layer):
    def __init__(self, attr_list):
        super(LinearModel, self).__init__()
        self._linear1 = paddle.nn.Linear(
            50, 30, weight_attr=attr_list[0], bias_attr=False)
        self._linear2 = paddle.nn.Linear(
            30, 10, weight_attr=attr_list[1], bias_attr=False)
        self._linear3 = paddle.nn.Linear(
            10, 10, weight_attr=attr_list[2], bias_attr=False)

    def forward(self, x):
        output = self._linear1(x)
        output = self._linear2(output)
        output = self._linear3(output)
        return output


class TestDistTraning(unittest.TestCase):
    def test_multiple_gpus(self):
        dist.init_parallel_env()
        process_group = init_process_group()

        dev_id = ParallelEnv().dev_id
        np.random.seed(2022 + dev_id)

        w_1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(
            np.random.normal(
                0, 0.02, size=(50, 30))))
        w_2 = paddle.ParamAttr(initializer=NumpyArrayInitializer(
            np.random.normal(
                0, 0.02, size=(30, 10))))
        w_3 = paddle.ParamAttr(initializer=NumpyArrayInitializer(
            np.random.normal(
                0, 0.02, size=(10, 10))))

        attr_list = [w_1, w_2, w_3]
        inp = np.random.rand(10, 50)

        # original reducer
        params_a = self.model_train(attr_list, inp, process_group=None)

        # refactored reducer in eager mode
        with _test_eager_guard():
            params_b = self.model_train(
                attr_list, inp, process_group=process_group)

        for i in range(len(params_a)):
            np.testing.assert_allclose(
                params_a[i].numpy(), params_b[i].numpy(), rtol=1e-6, atol=1e-6)

    def model_train(self, attr_list, inp, process_group=None):
        model = LinearModel(attr_list)
        model = paddle.DataParallel(model)
        optimizer = SGD(learning_rate=0.0003, parameters=model.parameters())

        x = paddle.to_tensor(inp, dtype="float32")
        x.stop_gradient = False

        for step in range(10):
            y = model(x)
            loss = y.mean()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        return model.parameters()


if __name__ == '__main__':
    unittest.main()
