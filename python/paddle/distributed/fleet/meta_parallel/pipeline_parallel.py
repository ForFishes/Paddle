#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import time
import copy
import os

from types import MethodType

from numpy import prod
import numpy
import paddle
import paddle.fluid as fluid
from .meta_parallel_base import MetaParallelBase
from .pp_utils.utils import get_tensor_bytes
from .pp_utils import utils
from .parallel_layers.pp_layers import PipelineLayer
from ..utils.log_util import logger
from . import p2p
from . import schedule


def is_even(number):
    return number % 2 == 0


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        super(PipelineParallel, self).__init__(layers, hcg, strategy)

        self.use_pipe_parallel = self._hcg.get_pipe_parallel_world_size() > 1
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1

        self.recv_cache = None
        self.grad_tensors = None

        self.meta_buffer = None

        self.send_meta = True
        self.first_gradient_send = True

        self.current_loss = paddle.to_tensor(0.0)
        self.total_loss = None

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'labels': [],  # labels from batch input
            'outputs': [],  # activations
            'output_tensors': [],  # tensor object to preserve backward graph
        }

        self.micro_batches = self._strategy.pipeline_configs['accumulate_steps']
        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        # self.micro_batch_size = self._strategy.pipeline_configs['micro_batch_size']
        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = (self.stage_id == (self.num_stages - 1))
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.first_output_send = True
        self.first_gradient_send = True
        self.pipe_recv_buf = None

        if self.is_last_stage:
            self.loss_model = self._layers._loss_fn

        p2p.initialize_p2p_groups(hcg)

        self.loss = paddle.to_tensor([0], dtype="int32")
        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage:
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage:
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage:
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage:
                p2p.send(self.loss, self.next_stage)

    def is_first_stage(self):
        return self.is_first_stage

    def is_last_stage(self):
        return self.is_last_stage

    def _allocate_buffers(self, shapes, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape in shapes:
                buffer.append(paddle.zeros(shape), dtype="float32")
            buffers.append(buffer)
        return buffers

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(paddle.zeros(shape, dtype="float32"))
            # buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def train_batch(self, data):
        self.batch = data
        self.total_loss = None
        sched = schedule.TrainSchedule(
            micro_batches=self.micro_batches,
            stages=self.num_stages,
            stage_id=self.stage_id)
        self._exec_schedule(sched)
        # self.agg_train_loss = self._aggregate_total_loss()
        # return self.agg_train_loss

    def _reserve_pipe_buffers(self, num_buffers):
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def _exec_optimizer_step(self):
        pass

    def _exec_reduce_grads(self):
        pass

    def _exec_reduce_tied_grads(self):
        pass

    def _exec_load_micro_batch(self, buffer_id):
        # batch = self._next_batch()
        # logger.info("start _exec_load_micro_batch")

        # input: batch[0], lable: batch[1]
        # TODO(shenliang03): set data according buffer_id
        batch = self.batch
        if self.is_first_stage:
            loaded = None
            if paddle.is_tensor(batch[0]):
                loaded = batch[0]
                # loaded = batch[0].clone().to(self.device).detach()
                # loaded.requires_grad = loaded.is_floating_point()
            # else:
            # if isinstance(batch[0], tuple):
            #     # assert isinstance(batch[0], tuple)
            #     # Assume list or tuple
            #     loaded = []
            #     for x in batch[0]:
            #         # assert torch.is_tensor(x)
            #         mine = x.clone().detach()
            #         # mine.requires_grad = mine.is_floating_point()
            #         loaded.append(mine)
            #     loaded = tuple(loaded)
            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage:
            loaded = batch[1]
            # if paddle.is_tensor(batch[1]):
            # loaded = batch[1]
            # elif isinstance(batch[1], tuple):
            # if isinstance(batch[1], tuple):
            #     loaded = []
            #     for x in batch[1]:
            #         x = x.detach()
            #         loaded.append(x)
            #     loaded = tuple(loaded)
            self.pipe_buffers['labels'][buffer_id] = loaded

    def _exec_forward_pass(self, buffer_id):
        # if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
        #     inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        # # else:
        # inputs = self.pipe_buffers['inputs'][buffer_id].clone()
        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        # self._zero_grads(inputs)

        outputs = self._layers(inputs)
        # logger.info("start _exec_forward_pass")
        # print("outputs", outputs)
        self.pipe_buffers['outputs'][buffer_id] = outputs

        if self.is_last_stage:
            if self.loss_model is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                self.loss = self.loss_model(outputs, labels)
            else:
                self.loss = outputs

            # if isinstance(self.loss, torch.Tensor):
            if self.total_loss is None:
                self.total_loss = paddle.zeros_like(self.loss)
            self.total_loss += self.loss.detach()

            # else:
            #     if self.total_loss is None:
            #         self.total_loss = [torch.zeros_like(l) for l in self.loss]
            #     for idx, l in enumerate(self.loss):
            #         self.total_loss[idx] += l.detach()

    def _exec_backward_pass(self, buffer_id):
        pass

    def _send_tensor_meta(self, buffer, recv_stage):
        # send_bytes = 0
        # if isinstance(buffer, torch.Tensor):
        print("start _send_tensor_meta")

        if paddle.is_tensor(buffer):
            type_tensor = paddle.to_tensor([0], dtype="int32")
            # type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            print("set type_tensor ", type_tensor, recv_stage)
            buffer_shape = buffer.shape
            send_shape = paddle.to_tensor(buffer_shape, dtype="int32")
            # send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = paddle.to_tensor([len(buffer_shape)], dtype="int32")
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            # send_bytes += _tensor_bytes(buffer)
        # if isinstance(buffer, tuple):
        #     type_tensor = paddle.to_tensor([2], dtype="int32")
        #     p2p.send(type_tensor, recv_stage)
        #     count_tensor = paddle.to_tensor([len(buffer)])
        #     p2p.send(count_tensor, recv_stage)
        #     for idx, tensor in enumerate(buffer):
        #         # assert isinstance(tensor, torch.Tensor)
        #         send_shape = paddle.to_tensor(tensor.size())
        #         send_ndims = paddle.to_tensor([len(tensor.size())])
        #         p2p.send(send_ndims, recv_stage)
        #         p2p.send(send_shape, recv_stage)
        #:
        #else
        #    raise NotImplementedError(
        #        f'Could not send meta type {type(buffer)}')

    def _recv_tensor_meta(self, send_stage):
        print("start _recv_tensor_meta")
        type_tensor = paddle.to_tensor([0], dtype="int32")
        p2p.recv(type_tensor, send_stage)
        print(type_tensor)
        recv_type = type_tensor.numpy()[0]
        # logger.info("recv_dtype : %d" % (recv_type))

        if recv_type == 0:
            recv_ndims = paddle.to_tensor([0], dtype="int32")
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = paddle.to_tensor([1] * recv_ndims, dtype="int32")
            p2p.recv(recv_shape, send_stage)
            recv_shape = list(recv_shape.numpy())
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]
        # # List or tuple of tensors
        # if recv_type == 1 or recv_type == 2:
        #     # count_tensor = torch.LongTensor(data=[0]).to(self.device)
        #     count_tensor = paddle.to_tensor([0], dtype="int32")
        #     p2p.recv(count_tensor, send_stage)
        #     num_tensors = count_tensor.item()
        #     num_tensors = num_tensors.numpy()[0]
        #     recv_shapes = []
        #     for idx in range(num_tensors):
        #         recv_ndims = paddle.to_tensor([0])
        #         p2p.recv(recv_ndims, send_stage)
        #         # recv_ndims = recv_ndims.item()
        #         recv_ndims = recv_ndims.numpy()[0]
        #         recv_shape = paddle.to_tensor([1] * recv_ndims, dtype="int32")
        #         p2p.recv(recv_shape, send_stage)
        #         recv_shapes.append(list(recv_shape.numpy()))

        #     buffers = self._allocate_buffers(recv_shapes, num_buffers=1)[0]
        #     # Convert to tuples if requested.
        #     if recv_type == 2:
        #         buffers = tuple(buffers)
        #     return buffers

        else:
            raise NotImplementedError(
                f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        # print("_exec_send_activations output:", outputs)
        if paddle.is_tensor(outputs):
            p2p.send(outputs, self.next_stage)
        # if isinstance(outputs, tuple):
        #     for idx, buffer in enumerate(outputs):
        #         p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

    def _exec_recv_activations(self, buffer_id):
        recvd = None
        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        if paddle.is_tensor(self.pipe_recv_buf):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            # recvd = self.pipe_recv_buf.clone().detach()
            recvd = self.pipe_recv_buf
            # recvd.requires_grad = recvd.is_floating_point()
        # else:
        # if isinstance(self.pipe_recv_buf, tuple):
        #     # assert isinstance(self.pipe_recv_buf, tuple)
        #     recvd = [None] * len(self.pipe_recv_buf)
        #     for idx, buffer in enumerate(self.pipe_recv_buf):
        #         p2p.recv(buffer, self.prev_stage)
        #         recvd[idx] = buffer.clone().detach()

        #     recvd = tuple(recvd)
        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_optimizer_step(self, lr_kwargs=None):
        pass

    def _exec_send_grads(self, buffer_id):
        pass

    def _exec_recv_grads(self, buffer_id):
        pass

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        # For each step in the schedule
        for step_cmds in pipe_schedule:
            print("cmds: ", step_cmds)
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)],
                                              self)
                self._exec_instr(**cmd.kwargs)
