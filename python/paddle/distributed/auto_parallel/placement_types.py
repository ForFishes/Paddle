#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This API design is inspired by:
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/placement_types.py
# Git commit hash: 52e2b87d00ed527dc7f990d1a7a4c5498f99c513

from paddle.framework import core


class Placement:
    def is_shard(self, dim):
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self):
        return isinstance(self, Replicate)

    def is_partial(self):
        return isinstance(self, Partial)


class Shard(Placement):
    def __init__(self, dim):
        self.dim = dim

    def __eq__(self, other):
        return isinstance(other, Shard) and self.dim == other.dim

    def __hash__(self):
        return hash(self.dim)

    def __repr__(self):
        return f"Shard(dim={self.dim})"

    def __str__(self):
        return f"S(dim={self.dim})"


class Replicate(Placement):
    def __eq__(self, other):
        return isinstance(other, Replicate)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return "Replicate()"

    def __str__(self):
        return "R"


class Partial(Placement):
    def __eq__(self, other):
        return isinstance(other, Partial)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return "Partial()"

    def __str__(self):
        return "P"


# Part1: Shard attributes related APIs
class DistAttr(core.TensorDistAttr):
    """
    DistAttr specifies how tensors are distributed or sliced on ProcessMesh.

    Args:
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        sharding_specs(list[str|None]): The specification describing how to shard the Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
            >>> dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

            >>> print(dist_attr)

    """

    def __init__(self, mesh, sharding_specs):
        # 1. inputs checking
        if not isinstance(mesh, core.ProcessMesh):
            raise ValueError(
                "The mesh must be an instance of paddle.distributed.ProcessMesh."
            )
        if not isinstance(sharding_specs, list):
            raise ValueError("The sharding_specs must be an instance of list.")
        assert all(
            isinstance(dim_name, str) or dim_name is None
            for dim_name in sharding_specs
        ), 'The dimension name in sharding_specs must be an instance of str.'

        self._sharding_specs = sharding_specs
        dims_mapping = [
            mesh.dim_names.index(dim_name) if dim_name is not None else -1
            for dim_name in sharding_specs
        ]

        # 2. init core.TensorDistAttr
        core.TensorDistAttr.__init__(self)

        self.process_mesh = mesh
        self.dims_mapping = dims_mapping
        self.mark_annotated("process_mesh")
        self.mark_annotated("dims_mapping")

    @property
    def sharding_specs(self):
        """
        Get sharding_specs of the dist_attr
        Returns:
            list[str]: sharding_specs
        """
        return self._sharding_specs


# class DTensorSpec:
#     def __init__(
#         self,
#         mesh: core.ProcessMesh,
#         placements: Tuple[Placement, ...],
#         tensor_shape: Tuple[int, ...],
#     ) -> None:
#         self.mesh = mesh
#         self.placements = placements
#         self.tensor_shape = tensor_shape

#     def __eq__(self, other: object) -> bool:
#         return self.mesh == other.mesh and self.placements == other.placements

#     def __hash__(self) -> int:
#         return hash((self.mesh, self.placements))

#     @property
#     def num_shards(self) -> int:
#         """
#         Get the number of shards in this DTensorSpec.
#         """
#         num_shards = 1
#         for idx, placement in enumerate(self.placements):
#             if placement.is_shard():
#                 num_shards *= self.mesh.shape[idx]
#         return num_shards

#     @property
#     def sums(self) -> List[int]:
#         return [
#             idx
#             for idx, placement in enumerate(self.placements)
#             if placement.is_partial()
#         ]

#     @property
#     def dim_map(self) -> List[int]:
#         r = [-1] * self.ndim
#         for i, placement in enumerate(self.placements):
#             if placement.is_shard():
#                 shard_dim = cast(Shard, placement).dim
#                 if r[shard_dim] > -1:
#                     raise ValueError(
#                         f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
#                         " DTensor operator implementation does not support things like hybrid"
#                         " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
#                     )
#                 r[shard_dim] = i
#         return r
