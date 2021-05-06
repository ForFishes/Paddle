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

import paddle
import paddle.distributed as dist

_groups = None
_hcg = None


def initialize_p2p_groups(hcg):
    global _groups, _hcg
    _groups = [dist.new_group(ranks=group) for group in hcg.get_p2p_groups()]
    _hcg = hcg


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups

    src_stage = _hcg.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)

    # src_rank = _hcg.stage_to_global(stage_id=src_stage)
    dst_rank = _hcg.stage_to_global(stage_id=dest_stage)
    print("send: ", group, "dst_rank: ", dst_rank)

    dist.send(tensor, dst=dst_rank, group=group, use_calc_stream=False)

    # dist.broadcast(tensor, src_rank, group=group, use_calc_stream=False)
    # dist.wait(tensor, group=group)
    return
    # return dist.broadcast(tensor, src_rank, group=group)


def recv(tensor, src_stage, async_op=False):
    global _groups

    dest_stage = _hcg.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _hcg.stage_to_global(stage_id=src_stage)

    print("recv: ", group, "src_rank: ", src_rank)
    # dist.broadcast(tensor, src_rank, group=group, use_calc_stream=False)
    dist.recv(tensor, src_rank, group=group, use_calc_stream=False)
    # def recv(tensor, src=0, group=None, use_calc_stream=True):

    # dist.wait(tensor, group=group)
    return
    # return dist.broadcast(tensor, src_rank, group=group)


def barrier(stage_id):
    global _groups, _hcg
    group_id = _hcg.stage_to_global(stage_id=stage_id)
    if (dist.get_rank() >= 0):
        print("Barrier Group ID", group_id)
        print("Barrier Group", _hcg.p2p_groups[group_id])
    dist.barrier(group=_groups[group_id])
    if (dist.get_rank() >= 0):
        print("Exiting Barrier ", group_id)


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _hcg.get_pipe_parallel_world_size() - 1

    if (src_stage == first_stage and dest_stage == last_stage or
            dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id correspods to group[group_id-num_stages+1, group_id]
     '''
    group_id = _hcg.stage_to_global(stage_id=stage_id)

    return _groups[group_id]
