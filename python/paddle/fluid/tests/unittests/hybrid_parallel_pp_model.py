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

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset
import unittest
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

HIDDEN_DIM = 32
LAYERS = 8

# class SimpleLoss(fluid.dygraph.Layer):
#     def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1,
#                  np_fc2, mp_id):
#         super(SimpleMPNet, self).__init__()

#         if mp_id == 0:
#             init_fc1_data = np_fc1[:, :(inner_size // 2)]
#             init_fc2_data = np_fc2[:(inner_size // 2), :]
#         else:
#             init_fc1_data = np_fc1[:, (inner_size // 2):]
#             init_fc2_data = np_fc2[(inner_size // 2):, :]

#         self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
#             hidden_size,
#             inner_size,
#             weight_attr=paddle.framework.ParamAttr(
#                 initializer=paddle.nn.initializer.Assign(init_fc1_data)),
#             gather_output=False,
#             has_bias=True)

#         self.linear2 = fleet.meta_parallel.RowParallelLinear(
#             inner_size,
#             hidden_size,
#             weight_attr=paddle.framework.ParamAttr(
#                 initializer=paddle.nn.initializer.Assign(init_fc2_data)),
#             input_is_parallel=True,
#             has_bias=True)

#         self.linear3 = paddle.nn.Linear(
#             hidden_size,
#             output_size,
#             weight_attr=paddle.framework.ParamAttr(
#                 initializer=paddle.nn.initializer.Constant(0.0)),
#             bias_attr=paddle.framework.ParamAttr(
#                 initializer=paddle.nn.initializer.Constant(0.0)))

#         self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
#             vocab_size,
#             hidden_size,
#             weight_attr=paddle.nn.initializer.Constant(value=0.5))

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)
#         return x


def sequential_model():
    model = paddle.nn.Sequential(
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        paddle.nn.Linear(HIDDEN_DIM, 1), )
    return model


class TestDistPPTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 8
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {"accumulate_steps": 1}
        paddle.distributed.init_parallel_env()
        fleet.init(is_collective=True, strategy=strategy)

    def test_mp_model(self):
        batch_input = paddle.randn(shape=(1, HIDDEN_DIM), dtype="float32")
        pipe_model = sequential_model()
        sgd = paddle.optimizer.SGD(learning_rate=0.0003, parameters=[])
        pipe_model = PipelineLayer(
            layers=pipe_model,
            num_stages=self.pipeline_parallel_size,
            loss_fn=None)
        pipe_model = fleet.distributed_model(pipe_model)

        # return
        # for step in range(2):
        #     print("step:", step)
        #     data = paddle.randn([2, HIDDEN_DIM], dtype="float32")
        #     # print("data", data)
        #     pipe_model.train_batch((data, data))
        # if pipe_model.stage_id == 0 or pipe_model.stage_id == 1:
        #     pipe_input = batch_input.clone().detach()
        #     pipe_input = paddle.cast(pipe_input, 'float32')

        #     def data_gen():
        #         gen = True
        #         while gen:
        #             yield [pipe_input, 0]
        #             gen = False

        #     loader = paddle.io.DataLoader.from_generator(capacity=5)
        #     loader.set_batch_generator(data_gen)
        #     data_iter = iter(loader)
        # else:
        #     data_iter = None
        # return True


if __name__ == "__main__":
    unittest.main()
