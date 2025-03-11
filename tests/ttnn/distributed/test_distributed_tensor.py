# # SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

# import torch
# import pytest
# import ttnn
# from loguru import logger
# from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
# from models.utility_functions import nearest_32


# @pytest.mark.parametrize(
#     "mesh_device",
#     [
#         32,
#     ],
#     indirect=True,
# )
# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_direct_replicate_to_tensor_mesh(mesh_device, dtype):
#     torch.manual_seed(1234)

#     mapper = ttnn.ReplicateTensorToMesh(mesh_device)

#     torch_tensor = torch.randn(1, 1, 32, 256)
#     replicated_tensors = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         mesh_mapper = mapper,
#         device=mesh_device,
#     )

#     out_tensors = ttnn.get_device_tensors(replicated_tensors)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensors[0]), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass

# @pytest.mark.parametrize(
#     "mesh_device",
#     [
#         32,
#     ],
#     indirect=True,
# )
# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_replicate_to_tensor_mesh(mesh_device, dtype):
#     torch.manual_seed(1234)

#     torch_tensor = torch.randn(1, 1, 32, 256)
#     to_repl = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         device=mesh_device,
#     )

#     mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
#     replicated_tensors = ttnn.distribute_tensor(to_repl, mapper, mesh_device)
#     out_tensors = ttnn.get_device_tensors(replicated_tensors)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensors[0]), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass


# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_shard_to_tensor_mesh(mesh_device, dtype):
#     torch.manual_seed(1234)

#     torch_tensor = torch.randn(1, 1, 32, 256)
#     to_shard = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         device=mesh_device,
#     )

#     mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

#     shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))

#     out_tensor = ttnn.aggregate_as_tensor(shards)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensor), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass


# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_concat_to_tensor(mesh_device, dtype):
#     torch.manual_seed(1234)

#     torch_tensor = torch.randn(1, 1, 32, 256)
#     to_shard = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         device=mesh_device,
#     )

#     mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

#     composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

#     out_tensor = ttnn.aggregate_tensor(ttnn.distribute_tensor(to_shard, mapper, mesh_device), composer)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensor), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass


# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_concat_slice_to_tensor(mesh_device, dtype):
#     torch.manual_seed(1234)

#     torch_tensor = torch.randn(1, 1, 32, 256)
#     to_shard = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         device=mesh_device,
#     )

#     mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=3)

#     composer = ttnn.concat_mesh_to_tensor_composer(dim=3)

#     sharded_tensor = ttnn.distribute_tensor(to_shard, mapper, mesh_device)

#     shards = ttnn.get_device_tensors(sharded_tensor)

#     out_tensor = ttnn.aggregate_tensor(shards, composer)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensor), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass


# @pytest.mark.parametrize(
#     "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
# )
# @pytest.mark.parametrize(
#     "M, K, N",
#     [pytest.param(32, 64, 128), pytest.param(32, 128, 64)],
# )
# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_shard2d_to_tensor_mesh(M, K, N, dtype, mesh_shape, mesh_device):
#     torch.manual_seed(1234)

#     torch_tensor = torch.randn(1, 1, M, K)
#     core_grid = ttnn.CoreGrid(y=1, x=8)

#     # If K < N it's FF1-like test case, else FF2-like test case
#     shard_dim = (0, 3) if K < N else (3, 0)

#     K = K // mesh_shape[1] if K < N else K // mesh_shape[0]
#     N = N // mesh_shape[0] if K < N else N // mesh_shape[1]

#     sharded_mem_config = ttnn.create_sharded_memory_config(
#         shape=(M // core_grid.y, K // core_grid.x),
#         core_grid=core_grid,
#         strategy=ttnn.ShardStrategy.WIDTH,
#         orientation=ttnn.ShardOrientation.ROW_MAJOR,
#         use_height_and_width_as_shard_shape=True,
#     )

#     to_shard = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=sharded_mem_config if M == 32 else ttnn.DRAM_MEMORY_CONFIG,
#         device=mesh_device,
#     )

#     mapper = ttnn.shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

#     shards = ttnn.get_device_tensors(ttnn.distribute_tensor(to_shard, mapper, mesh_device))

#     ttnn.aggregate_as_tensor(shards)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(shards), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass


# @pytest.mark.parametrize(
#     "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
# )
# @pytest.mark.parametrize(
#     "M, K, N",
#     [pytest.param(32, 128, 64), pytest.param(32, 128, 64)],
# )
# @pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.float32])
# def test_concat2d_to_tensor(M, K, N, dtype, mesh_shape, mesh_device):
#     torch.manual_seed(1234)

#     torch_tensor = torch.randn(1, 1, M, K)
#     core_grid = ttnn.CoreGrid(y=1, x=8)

#     # If K < N it's FF1-like test case, else FF2-like test case
#     shard_dim = (0, 3) if K < N else (3, 0)
#     concat_dim = (3, 1) if K < N else (1, 3)

#     K = K // mesh_shape[1] if K < N else K // mesh_shape[0]
#     N = N // mesh_shape[0] if K < N else N // mesh_shape[1]

#     sharded_mem_config = ttnn.create_sharded_memory_config(
#         shape=(M // core_grid.y, K // core_grid.x),
#         core_grid=core_grid,
#         strategy=ttnn.ShardStrategy.WIDTH,
#         orientation=ttnn.ShardOrientation.ROW_MAJOR,
#         use_height_and_width_as_shard_shape=True,
#     )

#     to_shard = ttnn.from_torch(
#         torch_tensor,
#         dtype=dtype,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=sharded_mem_config if M == 32 else ttnn.DRAM_MEMORY_CONFIG,
#         device=mesh_device,
#     )

#     mapper = ttnn.shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape=mesh_shape, dims=shard_dim)

#     composer = ttnn.concat_2d_mesh_to_tensor_composer(mesh_device, dims=concat_dim, mesh_shape=mesh_shape)

#     out_tensor = ttnn.aggregate_tensor(ttnn.distribute_tensor(to_shard, mapper, mesh_device), composer)

#     out_pass, out_pcc = comp_pcc(ttnn.to_torch(out_tensor), torch_tensor, pcc=0.99)
#     logger.info(f"PCC value: {out_pcc}")
#     assert out_pass
