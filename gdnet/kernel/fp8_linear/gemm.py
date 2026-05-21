from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

_kernel_cache: dict = {}


def _to_cute(t: torch.Tensor, cutlass_dtype, assumed_align: int = 16) -> cute.Tensor:
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):  # type: ignore
        t = t.view(torch.int8)  # type: ignore
    ct = from_dlpack(t, assumed_align=assumed_align)
    ct.element_type = cutlass_dtype
    leading = next(i for i, s in enumerate(t.stride()) if s == 1)
    return ct.mark_layout_dynamic(leading_dim=leading)


class _Fp8GemmSM90:
    def __init__(
        self,
        tile_shape_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ):
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.BFloat16
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        self.atom_layout_mnk = (1, 1, 1)
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = 2
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        wgmma_m = 64
        accum_regs_per_thread = (
            wgmma_m * tile_shape_mn[1] // self.num_threads_per_warp_group
        )
        self.mma_register_requirement = accum_regs_per_thread + 48
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.occupancy = 1
        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.mma_barriers = [
            pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.num_mma_threads
            ),
            pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.num_mma_threads
            ),
        ]
        self.epi_barriers = [
            pipeline.NamedBarrier(
                barrier_id=3, num_threads=self.num_threads_per_warp_group
            ),
            pipeline.NamedBarrier(
                barrier_id=4, num_threads=self.num_threads_per_warp_group
            ),
        ]
        self.num_mcast_ctas_a = cluster_shape_mn[1]
        self.num_mcast_ctas_b = cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.tiled_mma = None
        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("tile M must be 64 or 128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("tile N must be 64, 128, or 256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,  # type: ignore
            self.b_dtype,  # type: ignore
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])  # type: ignore
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        is_cooperative = False
        self.epi_tile = self._compute_epi_tile(
            self.tile_shape_mnk, self.c_dtype, is_cooperative
        )
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(d)
        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_d, tma_tensor_d = self._make_tma_store_atoms_and_tensors(
            d,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(
            d,
            self.tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2  # type: ignore
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[  # type: ignore
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[  # type: ignore
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[  # type: ignore
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel.set_name_prefix("fp8_gemm_sm90")
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            scale_a,
            scale_b,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mn: cute.Tensor,
        scale_a: cute.Tensor,
        scale_b: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        local_tidx = tidx % self.num_threads_per_warp_group

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_d)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))  # type: ignore
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))  # type: ignore
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)  # type: ignore
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        consumer_arrive_cnt = (
            (self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1)
            * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,  # type: ignore
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),  # type: ignore
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sD = storage.sD.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),  # type: ignore
            (None, None),
        )
        gB_nk = cute.local_tile(
            mB_nk,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),  # type: ignore
            (None, None),
        )
        gD_mn = cute.local_tile(
            mD_mn,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),  # type: ignore
            (None, None),
        )

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)  # type: ignore
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)  # type: ignore
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2),
        )

        k_tile_cnt = cute.size(gA_mk, mode=[2])  # type: ignore
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx == 0

        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                tile_coord_mn = work_tile.tile_idx
                tAgA_mk = tAgA[(None, tile_coord_mn[0], None)]
                tBgB_nk = tBgB[(None, tile_coord_mn[1], None)]
                mainloop_producer_state.reset_count()
                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    tAgA_k = tAgA_mk[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    tBgB_k = tBgB_nk[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            consumer_wg_idx = warp_group_idx - self.num_dma_warp_groups

            thr_mma = tiled_mma.get_slice(local_tidx)

            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)

            tCgD = thr_mma.partition_C(gD_mn)
            acc_shape = tCgD.shape[:3]
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            if consumer_wg_idx == 1:
                if work_tile.is_valid_tile:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])  # type: ignore

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                self.c_dtype,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(local_tidx)
            tRS_sD = thr_copy_r2s.partition_D(sD)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)  # type: ignore

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_threads_per_warp_group,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,  # type: ignore
                producer_group=tma_store_producer_group,
            )

            scale_val = scale_a[0] * scale_b[0]  # type: ignore

            if consumer_wg_idx == 1:
                self.mma_barriers[0].arrive()

            while work_tile.is_valid_tile:
                tile_coord_mn = work_tile.tile_idx
                gD_mn_slice = gD_mn[(None, None, tile_coord_mn[0], tile_coord_mn[1])]

                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()

                if consumer_wg_idx == 0:
                    self.mma_barriers[0].arrive_and_wait()
                else:
                    self.mma_barriers[1].arrive_and_wait()

                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):  # type: ignore
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()

                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):  # type: ignore
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)

                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                if consumer_wg_idx == 0:
                    self.mma_barriers[1].arrive()
                else:
                    self.mma_barriers[0].arrive()

                for i in range(cute.size(accumulators)):  # type: ignore
                    accumulators[i] = accumulators[i] * scale_val

                tCgD_for_tma_partition = cute.zipped_divide(gD_mn_slice, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sD, 0, 2),
                    tCgD_for_tma_partition,
                )
                epi_tile_num = cute.size(tCgD_for_tma_partition, mode=[1])  # type: ignore
                epi_tile_shape = tCgD_for_tma_partition.shape[1]  # type: ignore
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape,
                    stride=(epi_tile_shape[1], 1),  # type: ignore
                )
                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num

                for epi_idx in cutlass.range_constexpr(epi_tile_num):  # type: ignore
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):  # type: ignore
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(  # type: ignore
                        tRS_sD, mode=[3]
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    if consumer_wg_idx == 0:
                        self.epi_barriers[0].arrive_and_wait()
                    else:
                        self.epi_barriers[1].arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    epi_store_warp = (
                        self.epi_store_warp_id
                        + consumer_wg_idx * self.num_warps_per_warp_group
                    )
                    if warp_idx == epi_store_warp:
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    if consumer_wg_idx == 0:
                        self.epi_barriers[0].arrive_and_wait()
                    else:
                        self.epi_barriers[1].arrive_and_wait()

                tile_sched.advance_to_next_work()
                if work_tile.is_valid_tile:
                    tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()

    @staticmethod
    def _compute_stages(
        tile_shape_mnk, a_dtype, b_dtype, epi_tile, c_dtype, smem_capacity, occupancy
    ):
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))  # type: ignore
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))  # type: ignore
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8  # type: ignore
            + cute.size(b_shape) * b_dtype.width // 8  # type: ignore
        )
        epi_stage = 4
        epi_bytes = cute.size(epi_tile) * c_dtype.width // 8 * epi_stage  # type: ignore
        mbar_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - (mbar_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _compute_epi_tile(tile_shape_mnk, c_dtype, is_cooperative):
        if is_cooperative:
            return (
                min(128, cute.size(tile_shape_mnk, mode=[0])),  # type: ignore
                min(32, cute.size(tile_shape_mnk, mode=[1])),  # type: ignore
            )
        n_perf = 64 if c_dtype.width == 8 else 32
        return (
            min(64, cute.size(tile_shape_mnk, mode=[0])),  # type: ignore
            min(n_perf, cute.size(tile_shape_mnk, mode=[1])),  # type: ignore
        )

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk,
        epi_tile,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage,
        c_dtype,
        c_layout,
        epi_stage,
    ):
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))  # type: ignore
        a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K  # type: ignore
        b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K  # type: ignore
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )
        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))  # type: ignore
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(epi_tile, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))  # type: ignore
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim
        )

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_d, epi_smem_layout_staged, epi_tile):
        smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))  # type: ignore
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_d,
            smem_layout,
            epi_tile,
        )

    @staticmethod
    def _compute_grid(
        d,
        tile_shape_mnk,
        cluster_shape_mn,
        max_active_clusters,
        swizzle_size,
        raster_along_m,
    ):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))  # type: ignore
        gd = cute.zipped_divide(d, tiler=c_shape)
        num_ctas_mn = gd[(0, (None, None))].shape  # type: ignore
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        num_ctas_mnl = (*num_ctas_mn, 1)  # type: ignore
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid


def _pick_tile(M: int, N: int) -> tuple[int, int]:
    ratio = M / N
    if ratio >= 4:
        if N % 128 == 0 and M % 128 == 0:
            return (128, 128)
        if N % 64 == 0 and M % 128 == 0:
            return (128, 64)
    if N % 256 == 0 and M % 128 == 0:
        return (128, 256)
    if N % 128 == 0 and M % 128 == 0:
        return (128, 128)
    if N % 64 == 0 and M % 64 == 0:
        return (64, 64)
    return (128, 128)


def _pick_swizzle(M: int, N: int, tile_mn: tuple[int, int]) -> tuple[int, bool]:
    ratio = M / N
    raster_along_m = ratio <= 1
    min_dim = min(M // tile_mn[0], N // tile_mn[1])
    if min_dim >= 6:
        swizzle = 8
    elif min_dim >= 3:
        swizzle = 4
    elif min_dim >= 2:
        swizzle = 2
    else:
        swizzle = 1
    return swizzle, raster_along_m


def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    inv_scale_a: float,
    inv_scale_b: float,
) -> torch.Tensor:
    assert torch.cuda.get_device_capability(a.device) >= (9, 0), (
        "fp8_gemm requires SM90+"
    )
    assert a.dtype == torch.float8_e4m3fn  # type: ignore
    assert b.dtype == torch.float8_e4m3fn  # type: ignore
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert K % 16 == 0, "K must be a multiple of 16 for FP8 TMA alignment"
    assert M % 64 == 0 and N % 64 == 0, "M and N must be multiples of 64"

    d = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)  # type: ignore

    scale_a = torch.tensor([inv_scale_a], dtype=torch.float32, device=a.device)  # type: ignore
    scale_b = torch.tensor([inv_scale_b], dtype=torch.float32, device=b.device)  # type: ignore

    a_cute = _to_cute(a, cutlass.Float8E4M3FN)
    b_cute = _to_cute(b, cutlass.Float8E4M3FN)
    d_cute = _to_cute(d, cutlass.BFloat16)
    scale_a_cute = _to_cute(scale_a, cutlass.Float32)
    scale_b_cute = _to_cute(scale_b, cutlass.Float32)

    tile_mn = _pick_tile(M, N)
    swizzle_size, raster_along_m = _pick_swizzle(M, N, tile_mn)
    key = (M, N, K, tile_mn, swizzle_size, raster_along_m)
    if key not in _kernel_cache:
        hw = cutlass.utils.HardwareInfo()
        max_clusters = hw.get_max_active_clusters(1)
        stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        gemm_obj = _Fp8GemmSM90(
            tile_shape_mn=tile_mn,
            swizzle_size=swizzle_size,
            raster_along_m=raster_along_m,
        )
        _kernel_cache[key] = (
            cute.compile(
                gemm_obj,
                a_cute,
                b_cute,
                d_cute,
                scale_a_cute,
                scale_b_cute,
                max_clusters,
                stream,
            ),
            max_clusters,
        )

    compiled, max_clusters = _kernel_cache[key]
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    compiled(a_cute, b_cute, d_cute, scale_a_cute, scale_b_cute, stream)
    return d
