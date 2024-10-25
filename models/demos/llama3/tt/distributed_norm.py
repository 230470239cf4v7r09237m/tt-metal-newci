import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.llama_ccl import tt_sharded_distributed_rmsnorm, tt_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, TG):
        self.norm = norm
        self.args = args

        if TG:
            core_grid_ln = (4, 8)
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = 8192 // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        else:
            norm_input_grid = args.dram_shard_core_grid_for_k(args.dim // args.num_devices)
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                (
                    args.tile_padded_batch_rows,
                    args.dim // args.num_devices // norm_input_grid.num_cores,
                ),
                norm_input_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.gather_out_mem_cfg = ttnn.create_sharded_memory_config(
                (
                    args.tile_padded_batch_rows,
                    args.dim // norm_input_grid.num_cores,
                ),
                norm_input_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        self.TG = TG

    def forward(self, x, mode):
        """Apply a norm, possibly gathering inputs if required."""
        TG = self.TG
        if TG:
            if mode == "decode":
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            if mode == "decode":
                x = ttnn.interleaved_to_sharded(x, self.gather_in_mem_cfg)
                x = ttnn.all_gather(
                    x, dim=3, num_links=1, topology=self.args.ccl_topology(), memory_config=input_mem_cfg
                )
            else:
                x = ttnn.all_gather(
                    x, dim=3, num_links=1, topology=self.args.ccl_topology(), memory_config=input_mem_cfg
                )
        elif mode == "decode":
            # Gathered norms will be sharded for decode mode, so single-chip should be too
            x = ttnn.interleaved_to_sharded(x, input_mem_cfg)

        # x sharded in decode mode here
        x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology())

        return x
