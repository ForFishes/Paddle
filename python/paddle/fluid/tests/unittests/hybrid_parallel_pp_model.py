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
from hybrid_parallel_pp_layer import AlexNetPipeDesc, AlexNet, AlexNetPipe
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn as nn


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + rank_id)


HIDDEN_DIM = 32
LAYERS = 8

# def sequential_model():
#     model = paddle.nn.Sequential(
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
#         paddle.nn.Linear(HIDDEN_DIM, 1), )
#     return model


class TestDistPPTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {"accumulate_steps": 2}
        paddle.distributed.init_parallel_env()
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        base_model = AlexNet(10)

        init_net = AlexNetPipe()
        pipe_model = PipelineLayer(
            layers=init_net.to_layers(),
            num_stages=self.model_parallel_size,
            loss_fn=nn.CrossEntropyLoss())

        optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                         parameters=pipe_model.parameters())

        pipe_model = fleet.distributed_model(pipe_model)
        optimizer = fleet.distributed_optimizer(optimizer)

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=5, drop_last=True)

        for step_id, data in enumerate(train_reader()):
            batch_size = len(data)
            x_data = np.array([x[0] for x in data]).astype('float32').reshape(
                batch_size, 1, 28, 28)
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            label.stop_gradient = True

            if step_id > 5:
                return

            if hcg.get_stage_id() == 0:
                loss = pipe_model.train_batch(img, optimizer=optimizer)
            elif hcg.get_stage_id() == hcg.get_pipe_parallel_world_size() - 1:
                loss = pipe_model.train_batch(label, optimizer=optimizer)
            else:
                loss = pipe_model.train_batch(None, optimizer=optimizer)

            # loss = pipe_model(img, label)
            print(loss)
            # pipe_input = batch_input.clone().detach()
            # pipe_input = paddle.cast(pipe_input, 'float32')            

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
        return True


if __name__ == "__main__":
    unittest.main()
