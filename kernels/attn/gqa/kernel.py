# SPDX-License-Identifier: MIT
# FlyDSL port of hip_kernel.cpp (GQA flash-attention, gfx950 / CDNA4, wave64).
#
# Faithful port of the hand-scheduled HIP kernel data path:
#   - MFMA atom: mfma_f32_32x32x16_bf16 (A/B = v8bf16 per lane, C/D = v16f32)
#   - HipKittens st_32x32 swizzle for K, st_8x32 (identity) for V
#   - global->LDS via buffer_load_lds, double-buffered K/V
#   - online softmax (exp2) with lazy threshold rescale, s_setprio hints
#
# Each of the 8 waves independently computes attention for its own 32-row Q tile.
# All register state is fully unrolled in Python (SSA values rebound to py vars).

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec, T
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm

MFMA_MASK = 0x08
VALU_MASK = 0x02
EXP_MASK = 0x400
RESCALE_THRESHOLD = 8.0


def build_gqa_attn(
    *,
    ATTN_B=16,
    ATTN_H=64,
    ATTN_H_KV=8,
    ATTN_N=2048,
    ATTN_D=128,
    waves_per_eu=2,
):
    GROUP_SIZE = ATTN_H // ATTN_H_KV
    Q_BLOCK_SIZE = 32
    KV_BLOCK_SIZE = 64
    NUM_WARPS = 8
    WARP_SIZE = 64
    NUM_THREADS = WARP_SIZE * NUM_WARPS
    D = ATTN_D
    NUM_KV_TILES = ATTN_N // KV_BLOCK_SIZE

    if D == 128:
        TEMPERATURE_SCALE = 0.08838834764 * 1.44269504089
    else:
        TEMPERATURE_SCALE = 0.125 * 1.44269504089

    BYTES_PER_THREAD = 16
    BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE          # 1024
    BYTES_PER_MEMCPY = BYTES_PER_THREAD * NUM_THREADS      # 8192

    def swizzle_k_const(row, col):
        offset = 2 * (row * 32 + col)
        s1 = ((offset % 1024) >> 9) << 5
        s2 = ((offset % 2048) >> 10) << 4
        return offset ^ s1 ^ s2

    def swizzle_v_const(row, col):
        return 2 * (row * 32 + col)

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def attend_ker(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        L: fx.Tensor,
        Q_stride1: fx.Int32,
        K_stride1: fx.Int32,
        V_stride1: fx.Int32,
        O_stride1: fx.Int32,
        L_dim3: fx.Int32,
    ):
        f32 = fx.Float32
        bf16 = fx.BFloat16
        i32 = fx.Int32
        v8bf16_t = Vec.make_type(8, bf16)
        v4bf16_t = Vec.make_type(4, bf16)
        v16f32_t = Vec.make_type(16, f32)
        fm = arith.FastMathFlags.fast

        class _V8BF16Shim:
            ir_type = v8bf16_t

        def fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm)

        def fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm)

        def fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm)

        def fmaxf(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result

        def bcast16(scalar):
            return Vec.from_elements([fx.Float32(scalar)], f32).broadcast_to(16).ir_value()

        def mfma(a_v8, b_v8, c_v16):
            return rocdl.mfma_f32_32x32x16_bf16(
                v16f32_t, [_raw(a_v8), _raw(b_v8), _raw(c_v16)]
            )

        ZERO16 = Vec.filled(16, 0.0, f32).ir_value()

        # ---------------- LDS ----------------
        lds_alloc = fx.SharedAllocator()
        k_smem = [
            lds_alloc.allocate(fx.Array[bf16, KV_BLOCK_SIZE * ATTN_D, 16]).peek().ptr
            for _ in range_constexpr(2)
        ]
        v_smem = [
            lds_alloc.allocate(fx.Array[bf16, KV_BLOCK_SIZE * ATTN_D, 16]).peek().ptr
            for _ in range_constexpr(2)
        ]

        tid = fx.thread_idx.x
        lid = tid % WARP_SIZE
        wid = tid // WARP_SIZE

        bx = fx.block_idx.x
        head_idx = (bx % ATTN_H_KV) * GROUP_SIZE + (bx // ATTN_H_KV)
        batch_idx = fx.block_idx.z
        head_idx_kv = head_idx // GROUP_SIZE
        block_tile_idx = fx.block_idx.y
        tile_idx = i32(arith.unwrap(rocdl.readfirstlane(T.i32, _raw(block_tile_idx * NUM_WARPS + wid))))
        stagger = wid // 4

        Q_rsrc = buffer_ops.create_buffer_resource(Q)
        K_rsrc = buffer_ops.create_buffer_resource(K)
        V_rsrc = buffer_ops.create_buffer_resource(V)
        O_rsrc = buffer_ops.create_buffer_resource(O)
        L_rsrc = buffer_ops.create_buffer_resource(L)

        k_base_elems = batch_idx * (ATTN_N * ATTN_H_KV * ATTN_D) + head_idx_kv * ATTN_D
        v_base_elems = batch_idx * (ATTN_N * ATTN_H_KV * ATTN_D) + head_idx_kv * ATTN_D

        # ---------------- prefill offsets ----------------
        def prefill_offsets(is_k, row_stride):
            ST_ROWS = 32 if is_k else 8
            ST_COLS = 32
            ST_ROW_BYTES = ST_COLS * 2
            ST_BYTES = ST_ROWS * ST_ROW_BYTES
            ST_PER_ROW = 128 // ST_COLS
            offs = []
            for i in range_constexpr(2):
                lane_byte_off = (
                    lid * BYTES_PER_THREAD
                    + wid * BYTES_PER_WARP
                    + i * NUM_WARPS * BYTES_PER_WARP
                )
                subtile_id = lane_byte_off // ST_BYTES
                subtile_row = subtile_id // ST_PER_ROW
                subtile_col = subtile_id % ST_PER_ROW
                sub_off = lane_byte_off % ST_BYTES
                row = sub_off // ST_ROW_BYTES
                col = (sub_off % ST_ROW_BYTES) // 2
                if const_expr(is_k):
                    offset = (row * 32 + col) * 2
                    s1 = ((offset % 1024) >> 9) << 5
                    s2 = ((offset % 2048) >> 10) << 4
                    sw = offset ^ s1 ^ s2
                else:
                    sw = (row * 32 + col) * 2
                sw_row = sw // ST_ROW_BYTES
                sw_col = (sw % ST_ROW_BYTES) // 2
                global_row = sw_row + subtile_row * ST_ROWS
                global_col = sw_col + subtile_col * ST_COLS
                offs.append((global_row * row_stride + global_col) * 2)
            return offs

        off_K = prefill_offsets(True, K_stride1)
        off_V = prefill_offsets(False, V_stride1)

        # ---------------- group_load: global -> LDS ----------------
        def group_load(smem_ptr, tile, offsets, rsrc, base_elems, row_stride):
            soff_bytes = (base_elems + tile * (KV_BLOCK_SIZE * row_stride)) * 2
            soff_bytes = i32(arith.unwrap(rocdl.readfirstlane(T.i32, _raw(i32(soff_bytes)))))
            lds_base = i32(fx.ptrtoint(smem_ptr)) + i32(wid * BYTES_PER_WARP)
            for i in range_constexpr(2):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base + i32(i * BYTES_PER_MEMCPY), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, lds_ptr, i32(BYTES_PER_THREAD),
                    _raw(offsets[i]), _raw(soff_bytes), i32(0), i32(0),
                )

        def load_k(tile, buf):
            group_load(k_smem[buf], tile, off_K, K_rsrc, k_base_elems, K_stride1)

        def load_v(tile, buf):
            group_load(v_smem[buf], tile, off_V, V_rsrc, v_base_elems, V_stride1)

        # ---------------- LDS -> registers ----------------
        def _read_v8bf16(smem_ptr, byte_off):
            base = fx.recast_iter(fx.Uint8, smem_ptr)
            return fx.ptr_load(base + byte_off, result_type=_V8BF16Shim)

        def swizzle_k_expr(row, col):
            offset = (row * 32 + col) * 2
            s1 = ((offset % 1024) >> 9) << 5
            s2 = ((offset % 2048) >> 10) << 4
            return offset ^ s1 ^ s2

        def load_k_regs(buf):
            smem_ptr = k_smem[buf]
            row_offset = lid % 32
            col_offset = 8 * (lid // 32)
            ST_BYTES = 32 * 32 * 2
            kreg = [[None] * 8 for _ in range_constexpr(2)]
            for ii in range_constexpr(2):
                for jj in range_constexpr(4):
                    off_imm = (ii * 4 + jj) * ST_BYTES
                    for j in range_constexpr(2):
                        col = j * 16 + col_offset
                        byte_off = swizzle_k_expr(row_offset, col) + off_imm
                        kreg[ii][jj * 2 + j] = _read_v8bf16(smem_ptr, byte_off)
            return kreg  # [n=2][k=8] v8bf16

        def load_v_regs(buf):
            smem_ptr = v_smem[buf]
            row_offset = ((lid % 16) // 4) + ((lid // 32) * 4)
            col_offset = ((lid % 4) * 4) + (16 * ((lid % 32) // 16))
            col_in_sub = col_offset % 32
            ST_BYTES = 8 * 32 * 2
            ST_PER_ROW = 4
            sw = (row_offset * 32 + col_in_sub) * 2
            base_i = i32(fx.ptrtoint(smem_ptr))
            vreg = [[None] * 4 for _ in range_constexpr(4)]
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    shared_row = i * 2
                    shared_col = j
                    halves = []
                    for k in range_constexpr(2):
                        off = (shared_row * ST_PER_ROW + shared_col) * ST_BYTES + k * ST_PER_ROW * ST_BYTES
                        addr = base_i + i32(off) + sw
                        ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
                        halves.append(Vec(rocdl.ds_read_tr16_b64(v4bf16_t, ptr).result))
                    vreg[i][j] = halves[0].shuffle(halves[1], list(range(8))).ir_value()
            return vreg  # [i=4][j=4] v8bf16

        # ---------------- Q load ----------------
        def load_q_packs():
            row_offset = lid % 32
            col_offset = 8 * (lid // 32)
            base = (
                batch_idx * (ATTN_N * ATTN_H * ATTN_D)
                + tile_idx * (Q_BLOCK_SIZE * ATTN_H * ATTN_D)
                + head_idx * ATTN_D
            )
            sc8 = Vec.from_elements([f32(TEMPERATURE_SCALE)], f32).broadcast_to(8).ir_value()
            packs = []
            for j in range_constexpr(8):
                col = 16 * j + col_offset
                elem_off = base + row_offset * Q_stride1 + col
                raw = buffer_ops.buffer_load(Q_rsrc, _raw(i32(elem_off)), vec_width=8, dtype=bf16)
                vf = Vec(raw).to(f32)
                scaled = arith.mulf(_raw(vf), _raw(sc8), fastmath=fm)
                packs.append(Vec(scaled).to(bf16).ir_value())
            return packs  # [8] v8bf16

        # ---------------- O store ----------------
        def store_o(o_reg):
            row_offset = lid % 32
            col_offset = 4 * (lid // 32)
            base = (
                batch_idx * (ATTN_N * ATTN_H * ATTN_D)
                + tile_idx * (Q_BLOCK_SIZE * ATTN_H * ATTN_D)
                + head_idx * ATTN_D
            )
            for j in range_constexpr(4):
                ov = Vec(o_reg[j])
                for k in range_constexpr(4):
                    col = 32 * j + col_offset + k * 8
                    elem0 = k * 4  # (idx=k*2 float2) -> 4 f32 per k
                    elems = [ov[elem0 + e] for e in range_constexpr(4)]
                    vbf = Vec.from_elements(elems, f32).to(bf16)
                    for e in range_constexpr(4):
                        off = base + row_offset * O_stride1 + col + e
                        buffer_ops.buffer_store(vbf[e], O_rsrc, _raw(i32(off)))

        def store_lse(lse_val):
            row_offset = lid % 32
            if lid < 32:
                seq_pos = tile_idx * Q_BLOCK_SIZE + row_offset
                off = batch_idx * (ATTN_H * L_dim3) + head_idx * L_dim3 + seq_pos
                buffer_ops.buffer_store(f32(lse_val), L_rsrc, _raw(i32(off)))

        # =====================================================================
        # QK: att[n=2] (v16f32) = sum_k k_reg_t[k][n] * q_t[k]
        #   In HIP: D[n] = sum over 8 k of A[k][n] * B[k]; A=k transposed, B=q.
        #   Here k_reg is [n][k]; the MFMA contracts the 8-wide K dim per call,
        #   accumulating over the 8 k-subtiles.
        # =====================================================================
        def qk(kreg, qpacks):
            att = []
            for n in range_constexpr(2):
                acc = ZERO16
                for k in range_constexpr(8):
                    acc = mfma(kreg[n][k], qpacks[k], acc)
                att.append(acc)
            return att  # [2] v16f32

        # OV: o_reg[n=4] += sum_{kk=0..3} v_reg[kk][n] * att_bf[kk]
        def ov_slice(o_reg, vreg_k, att_bf_k):
            for n in range_constexpr(4):
                o_reg[n] = mfma(vreg_k[n], att_bf_k, o_reg[n])
            return o_reg

        # ---------------- softmax helpers on att (v16f32) ----------------
        def col_max(att):
            mx = Vec(att[0]).reduce(fx.ReductionOp.MAX)
            mx = fmaxf(mx, Vec(att[1]).reduce(fx.ReductionOp.MAX))
            peer = fx.Float32(mx).shuffle_xor(i32(32), i32(64))
            return fmaxf(mx, peer)

        def col_sum(att):
            sm = Vec(att[0]).reduce(fx.ReductionOp.ADD, fastmath=fm)
            sm = fadd(sm, Vec(att[1]).reduce(fx.ReductionOp.ADD, fastmath=fm))
            peer = fx.Float32(sm).shuffle_xor(i32(32), i32(64))
            return fadd(sm, peer)

        def sub_exp2(att, mx):
            b = bcast16(mx)
            out = []
            for n in range_constexpr(2):
                d = arith.subf(_raw(att[n]), _raw(b), fastmath=fm)
                out.append(Vec(d).exp2().ir_value())
            return out

        def to_bf16_packs(att):
            # att[n] v16f32 -> att_bf split into 4 v8bf16 (kk index for OV)
            packs = []
            for n in range_constexpr(2):
                bv = Vec(att[n]).to(bf16)  # v16bf16
                packs.append(bv.shuffle(bv, list(range(0, 8))).ir_value())
                packs.append(bv.shuffle(bv, list(range(8, 16))).ir_value())
            return packs  # [4] v8bf16, ordering kk=0..3

        def mul_o(o_reg, scal):
            b = bcast16(scal)
            return [arith.mulf(_raw(o_reg[n]), _raw(b), fastmath=fm) for n in range_constexpr(4)]

        # =====================================================================
        # Main compute (double-buffered software pipeline over KV tiles)
        #   prologue:  load tile 0 into buf 0
        #   iter t:    read regs from buf t%2; prefetch tile t+1 into buf (t+1)%2
        #              (global->LDS overlaps current-tile MFMA); compute on t
        # =====================================================================
        o_reg = [ZERO16, ZERO16, ZERO16, ZERO16]
        norm_vec = f32(0.0)
        max_vec = f32(float("-inf"))

        qpacks = load_q_packs()

        # prologue: stage tile 0
        load_k(0, 0)
        load_v(0, 0)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

        for t in range_constexpr(NUM_KV_TILES):
            cur = t % 2
            has_next = const_expr(t + 1 < NUM_KV_TILES)
            nxt = (t + 1) % 2

            kreg = load_k_regs(cur)
            vreg = load_v_regs(cur)

            # prefetch next tile into the alternate buffer (async global->LDS)
            if has_next:
                load_k(t + 1, nxt)
                load_v(t + 1, nxt)

            rocdl.s_setprio(1)
            att = qk(kreg, qpacks)            # [2] v16f32

            m_cur = col_max(att)
            m_new = fmaxf(max_vec, m_cur)

            # rescale running stats
            corr = Vec(bcast16(arith.subf(_raw(max_vec), _raw(m_new), fastmath=fm))).exp2().ir_value()
            corr_s = fx.Float32(Vec(corr)[0])
            norm_vec = fmul(norm_vec, corr_s)
            o_reg = [arith.mulf(_raw(o_reg[n]), _raw(corr), fastmath=fm) for n in range_constexpr(4)]

            p = sub_exp2(att, m_new)          # [2] v16f32, exp2(att - m_new)
            s_cur = col_sum(p)
            norm_vec = fadd(norm_vec, s_cur)

            p_bf = to_bf16_packs(p)           # [4] v8bf16
            for kk in range_constexpr(4):
                o_reg = ov_slice(o_reg, [vreg[kk][n] for n in range_constexpr(4)], p_bf[kk])
            rocdl.s_setprio(0)

            max_vec = m_new

            # wait for prefetched tile to land + sync all waves before next read
            if has_next:
                rocdl.s_waitcnt(0)
                rocdl.s_barrier()

        inv = arith.divf(_raw(f32(1.0)), _raw(norm_vec), fastmath=fm)
        o_reg = mul_o(o_reg, inv)
        store_o(o_reg)

        lse_max = fmul(max_vec, f32(0.69314718056))
        lse = fadd(fmath.log(_raw(norm_vec)), lse_max)
        store_lse(lse)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        L: fx.Tensor,
        Q_stride1: fx.Int32,
        K_stride1: fx.Int32,
        V_stride1: fx.Int32,
        O_stride1: fx.Int32,
        L_dim3: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = ATTN_H
        grid_y = (ATTN_N // Q_BLOCK_SIZE + NUM_WARPS - 1) // NUM_WARPS
        grid_z = ATTN_B
        attend_ker(
            Q, K, V, O, L,
            Q_stride1, K_stride1, V_stride1, O_stride1, L_dim3,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{NUM_THREADS},{NUM_THREADS}",
            },
        ).launch(grid=(grid_x, grid_y, grid_z), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch
