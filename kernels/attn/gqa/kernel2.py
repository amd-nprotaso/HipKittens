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
from flydsl._mlir.dialects import scf as _scf

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

        # ---------------- group_load: global -> LDS ----------------
        # lds_base is a wave-uniform SGPR value (readfirstlane-hoisted), matching
        # k_lds_base_* / v_lds_base_* in hip_kernel.cpp.
        def group_load(lds_base, tile, offsets, rsrc, base_elems, row_stride):
            soff_bytes = (base_elems + tile * (KV_BLOCK_SIZE * row_stride)) * 2
            soff_bytes = i32(arith.unwrap(rocdl.readfirstlane(T.i32, _raw(i32(soff_bytes)))))
            for i in range_constexpr(2):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base + i32(i * BYTES_PER_MEMCPY), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, lds_ptr, i32(BYTES_PER_THREAD),
                    _raw(offsets[i]), _raw(soff_bytes), i32(0), i32(0),
                )

        def load_k(tile, buf):
            group_load(k_lds_base[buf], tile, off_K, k_rsrc[buf], k_base_elems, K_stride1)

        def load_v(tile, buf):
            group_load(v_lds_base[buf], tile, off_V, v_rsrc[buf], v_base_elems, V_stride1)

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
            O_rsrc = buffer_ops.create_buffer_resource(O)
            for j in range_constexpr(4):
                ov = Vec(o_reg[j])
                for k in range_constexpr(4):
                    col = 32 * j + col_offset + k * 8
                    elem0 = k * 4  # (idx=k*2 float2) -> 4 f32 per k
                    elems = [ov[elem0 + e] for e in range_constexpr(4)]
                    # Vectorized store: 4 bf16 as one buffer_store_b64 (mirrors HIP
                    # store_o_global's buffer_store_b64), instead of 4 scalar shorts.
                    vbf = Vec.from_elements(elems, f32).to(bf16)
                    off = base + row_offset * O_stride1 + col
                    buffer_ops.buffer_store(vbf.ir_value(), O_rsrc, _raw(i32(off)))

        def store_lse(lse_val):
            row_offset = lid % 32
            if lid < 32:
                L_rsrc = buffer_ops.create_buffer_resource(L)
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

        # mma_AtB_QK (mirrors hip_kernel.cpp:177):
        #   D[n] = C[n] + sum_{k=0..7} A[k][n] * B[k][0]
        #   A = k_reg_t [8][2] v8bf16, B = q_reg_t [8][1] v8bf16, C = [2] v16f32.
        def mma_AtB_QK(A, B, C):
            D = [None, None]
            for n in range_constexpr(2):
                acc = mfma(A[0][n], B[0][0], C[n])
                for k in range_constexpr(1, 8):
                    acc = mfma(A[k][n], B[k][0], acc)
                D[n] = acc
            return D  # [2] v16f32

        # OV: o_reg[n=4] += sum_{kk=0..3} v_reg[kk][n] * att_bf[kk]
        def ov_slice(o_reg, vreg_k, att_bf_k):
            for n in range_constexpr(4):
                o_reg[n] = mfma(vreg_k[n], att_bf_k, o_reg[n])
            return o_reg

        # mma_AtB_OV_slice (mirrors hip_kernel.cpp:216): one OV contraction slice.
        #   D[n] += A[n] * B  for n in 0..3
        #   A = v_reg[k] ([4] v8bf16), B = att_block_bf16[k] (v8bf16).
        def mma_AtB_OV_slice(D, A, B):
            for n in range_constexpr(4):
                D[n] = mfma(A[n], B, D[n])
            return D

        # mma_AtB_OV (mirrors hip_kernel.cpp:198): full OV multiply.
        #   D[n] = C[n] + sum_{k=0..3} A[k][n] * B[k]
        #   A = v_reg [4][4] v8bf16, B = att_block_bf16 [4] v8bf16, C = o_reg [4] v16f32.
        def mma_AtB_OV(C, A, B):
            D = [None] * 4
            for n in range_constexpr(4):
                acc = mfma(A[0][n], B[0], C[n])
                for k in range_constexpr(1, 4):
                    acc = mfma(A[k][n], B[k], acc)
                D[n] = acc
            return D  # [4] v16f32

        # ---------------- softmax helpers on att (v16f32) ----------------
        def col_max(att):
            mx = Vec(att[0]).reduce(fx.ReductionOp.MAX)
            mx = fmaxf(mx, Vec(att[1]).reduce(fx.ReductionOp.MAX))
            peer = fx.Float32(mx).shuffle_xor(i32(32), i32(64))
            return fmaxf(mx, peer)

        # lane_below (hip_kernel.cpp:286): per-lane (cur - prev) <= T -> i1.
        def lane_below(prev, cur, thr):
            delta = arith.subf(_raw(cur), _raw(prev), fastmath=fm)
            return arith.cmpf(arith.CmpFPredicate.OLE, _raw(delta), _raw(f32(thr)))

        # wave_all_ok (hip_kernel.cpp:290): __all(lane_ok) -> i1, true iff every
        # active lane is below threshold. __all(x) == (ballot(!x) == 0).
        _i1_t = ir.IntegerType.get_signless(1)

        def wave_all_ok(lane_ok):
            true_c = arith.ConstantOp(_i1_t, ir.IntegerAttr.get(_i1_t, 1)).result
            not_ok = arith.XOrIOp(_raw(lane_ok), true_c).result
            mask = rocdl.ballot(T.i64, not_ok)
            return arith.cmpi(arith.CmpIPredicate.eq, _raw(mask), _raw(fx.Int64(0)))

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

        # ---- finer-grained softmax pieces (match reference cluster splitting) ----
        def col_sum_into(att, norm):
            return fadd(norm, col_sum(att))

        def sub_col(att, mx):
            b = bcast16(mx)
            return [arith.subf(_raw(att[n]), _raw(b), fastmath=fm) for n in range_constexpr(2)]

        def exp2_one(v16):
            return Vec(v16).exp2().ir_value()

        # ---- scheduling helpers (mirror reference sched_barrier templates) ----
        def sb0():
            rocdl.sched_barrier(0)

        def bar():
            rocdl.s_barrier()

        def sched_pairs(pairs, valu_cnt, group):
            for _p in range_constexpr(pairs):
                rocdl.sched_group_barrier(MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(VALU_MASK, valu_cnt, group)

        def sched_exp_pairs(pairs, exp_cnt, group):
            for _p in range_constexpr(pairs):
                rocdl.sched_group_barrier(MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(EXP_MASK, exp_cnt, group)

        # s_waitcnt via the ROCDL intrinsic (not inline asm). The AMDGPU
        # SIInsertWaitcnts backend pass understands rocdl.s_waitcnt and folds it
        # into its own analysis, so it does NOT re-insert conservative waits on
        # top (vmcnt(0) full-drains / extra partial lgkmcnt) the way it does for
        # an opaque inline-asm blob. gfx950 s_waitcnt bitfield encoding, matching
        # kernels/flash_attn_gfx950.py:
        #   vmcnt = bits[3:0] | (bits[15:14] << 4); expcnt = bits[6:4];
        #   lgkmcnt = bits[13:8].
        _VMCNT_LO_MASK = 0xF
        _LGKMCNT_EXPCNT_BASE = 0x3F70  # vmcnt=0, expcnt=7(max), lgkmcnt=63(max)
        _VMCNT_HI_SHIFT = 14
        _VMCNT_HI_MASK = 0x3
        _LGKMCNT_0_ONLY = 0xC07F  # vmcnt=63(max), expcnt=7(max), lgkmcnt=0

        def wait_vmcnt(n):
            # vmcnt(n) only; leave lgkmcnt/expcnt maxed (no wait on those).
            val = (
                (n & _VMCNT_LO_MASK)
                | _LGKMCNT_EXPCNT_BASE
                | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
            )
            rocdl.s_waitcnt(val)

        def wait_lgkmcnt0():
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)

        # ---- lazy-threshold rescale (reference lane_below + __all vote) ----
        # Returns (corr_vec16, corr_scalar, kept_max). corr == 1.0 when every lane is
        # within RESCALE_THRESHOLD of the running max (skip rescale); else exp2(prev-new).
        def lazy_rescale(att, max_prev):
            m_cur = col_max(att)
            m_new = fmaxf(m_cur, max_prev)
            delta = arith.subf(_raw(m_new), _raw(max_prev), fastmath=fm)
            not_ok = arith.cmpf(arith.CmpFPredicate.OGT, _raw(delta), _raw(f32(RESCALE_THRESHOLD)))
            mask = rocdl.ballot(T.i64, not_ok)
            all_ok = arith.cmpi(arith.CmpIPredicate.eq, _raw(mask), _raw(fx.Int64(0)))
            kept_max = arith.select(all_ok, _raw(max_prev), _raw(m_new))
            scale = arith.subf(_raw(max_prev), _raw(m_new), fastmath=fm)
            scale_exp = Vec(bcast16(scale)).exp2().ir_value()
            corr_s = arith.select(all_ok, _raw(f32(1.0)), _raw(Vec(scale_exp)[0]))
            corr_v = bcast16(corr_s)
            return corr_v, fx.Float32(corr_s), fx.Float32(kept_max)

        # =====================================================================
        # stage_qk: QK[tile] + start of softmax (max update + sub both heights +
        #   exp2 of height 0). Returns the partially-softmaxed att ([att_h0_exp,
        #   att_h1_subbed]), the kept max, and rescaled (o_reg, norm). The o_reg /
        #   norm rescale here corresponds to the reference's lazy-threshold vote;
        #   it must be emitted AFTER the previous tile's OV has accumulated.
        # =====================================================================
        def stage_qk(kreg, max_prev, o_reg, norm_vec, first):
            att = qk(kreg, qpacks)                       # [2] v16f32
            if const_expr(first):
                max_new = fx.Float32(col_max(att))       # reference col_max_reset
                corr_applied_o = o_reg
                corr_applied_n = norm_vec
            else:
                corr_v, corr_s, max_new = lazy_rescale(att, max_prev)
                corr_applied_o = [
                    arith.mulf(_raw(o_reg[n]), _raw(corr_v), fastmath=fm) for n in range_constexpr(4)
                ]
                corr_applied_n = fmul(norm_vec, corr_s)
            sub = sub_col(att, max_new)                  # [2] v16f32 (both heights)
            sub[0] = exp2_one(sub[0])                    # finish height 0 only
            return sub, max_new, corr_applied_o, corr_applied_n

        # stage_finish: exp2 height 1, accumulate col_sum into norm, OV with vreg.
        def stage_finish(att_partial, vreg, o_reg, norm_vec):
            att = [att_partial[0], exp2_one(att_partial[1])]
            norm_vec = col_sum_into(att, norm_vec)
            p_bf = to_bf16_packs(att)                    # [4] v8bf16
            for kk in range_constexpr(4):
                o_reg = ov_slice(o_reg, [vreg[kk][n] for n in range_constexpr(4)], p_bf[kk])
            return o_reg, norm_vec

        # ---------------- LDS ----------------
        # Matches hip_kernel.cpp: contiguous k_smem_0, k_smem_1, v_smem_0, v_smem_1.
        lds_alloc = fx.SharedAllocator()
        k_smem_0 = lds_alloc.allocate(fx.Array[bf16, KV_BLOCK_SIZE * ATTN_D, 16]).peek().ptr
        k_smem_1 = lds_alloc.allocate(fx.Array[bf16, KV_BLOCK_SIZE * ATTN_D, 16]).peek().ptr
        v_smem_0 = lds_alloc.allocate(fx.Array[bf16, KV_BLOCK_SIZE * ATTN_D, 16]).peek().ptr
        v_smem_1 = lds_alloc.allocate(fx.Array[bf16, KV_BLOCK_SIZE * ATTN_D, 16]).peek().ptr
        k_smem = [k_smem_0, k_smem_1]
        v_smem = [v_smem_0, v_smem_1]

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

        # Per-warp LDS destination bases, hoisted into SGPRs (wave-uniform), matching
        # k_lds_base_* / v_lds_base_* in hip_kernel.cpp:
        #   readfirstlane((uint32_t)smem + warpid() * BYTES_PER_WARP)
        lds_warp_off = i32(wid * BYTES_PER_WARP)

        def _lds_base(smem_ptr):
            #base = i32(fx.ptrtoint(smem_ptr)) + lds_warp_off
            return i32(arith.unwrap(rocdl.readfirstlane(T.i32, _raw(i32(fx.ptrtoint(smem_ptr)) + lds_warp_off))))

        k_lds_base = [_lds_base(k_smem_0), _lds_base(k_smem_1)]
        v_lds_base = [_lds_base(v_smem_0), _lds_base(v_smem_1)]


        k_base_elems = batch_idx * (ATTN_N * ATTN_H_KV * ATTN_D) + head_idx_kv * ATTN_D
        v_base_elems = batch_idx * (ATTN_N * ATTN_H_KV * ATTN_D) + head_idx_kv * ATTN_D

        # Single base descriptor per tensor (mirrors k_srsrc_base / v_srsrc_base in
        # hip_kernel.cpp); both double-buffer slots share the same resource.
        k_srsrc_base = buffer_ops.create_buffer_resource(K)
        v_srsrc_base = buffer_ops.create_buffer_resource(V)
        Q_rsrc = buffer_ops.create_buffer_resource(Q)
        k_rsrc = [k_srsrc_base, k_srsrc_base]
        v_rsrc = [v_srsrc_base, v_srsrc_base]

        # =====================================================================
        # Register tiles (SSA equivalents of hip_kernel.cpp's register arrays).
        # In FlyDSL each tile is a nested Python list of SSA values:
        #   innermost [...][8] float2  -> one v16f32  (16 packed f32)
        #   innermost [...][4] bf16_2  -> one v8bf16   (8 packed bf16)
        # Layouts mirror the reference exactly.
        # =====================================================================
        ZERO16 = Vec.filled(16, 0.0, f32).ir_value()

        # bf16 q_reg[1][8][8]      -> [1][8] v8bf16
        q_reg = [[None] * 8 for _ in range_constexpr(1)]
        # bf16_2 q_reg_t[8][1][4]  -> [8][1] v8bf16
        q_reg_t = [[None] * 1 for _ in range_constexpr(8)]
        # bf16_2 k_reg[2][8][4]    -> [2][8] v8bf16
        k_reg = [[None] * 8 for _ in range_constexpr(2)]
        # bf16_2 k_reg_t[8][2][4]  -> [8][2] v8bf16
        k_reg_t = [[None] * 2 for _ in range_constexpr(8)]
        # bf16_2 v_reg[4][4][4]    -> [4][4] v8bf16
        v_reg = [[None] * 4 for _ in range_constexpr(4)]
        # float2 o_reg[4][8]       -> [4] v16f32 (zero-initialized accumulator)
        o_reg = [ZERO16, ZERO16, ZERO16, ZERO16]
        # float2 att_block[2][2][8] -> [2 double-buf][2 height] v16f32
        att_block = [[None] * 2 for _ in range_constexpr(2)]
        # bf16_2 att_block_bf16[2][8] -> [2] v8bf16
        att_block_bf16 = [None] * 2

        # scalar softmax accumulators (max_vec, norm_vec, scale_vec)
        max_vec = f32(float("-inf"))
        max_vec_prev = max_vec
        norm_vec = f32(0.0)
        scale_vec = f32(1.0)

        # Prefill swizzled global->LDS offsets (mirrors hip_kernel.cpp:
        #   prefill_offsets<true>(swizzled_offsets_K, K_stride1);
        #   prefill_offsets<false>(swizzled_offsets_V, V_stride1);)
        swizzled_offsets_K = prefill_offsets(True, K_stride1)
        swizzled_offsets_V = prefill_offsets(False, V_stride1)
        off_K = swizzled_offsets_K
        off_V = swizzled_offsets_V

        # ---------------- Load K[0] into shared ----------------
        # Mirrors hip_kernel.cpp:
        #   group_load_srsrc<true>(0, ...);  // K[0] -> k_smem_0
        #   __builtin_amdgcn_s_waitcnt(0);
        #   __builtin_amdgcn_sched_barrier(0);
        #   __builtin_amdgcn_s_barrier();
        load_k(0, 0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # ---------------- Load Q into registers ----------------
        # Mirrors hip_kernel.cpp:
        #   float2 q_reg_fl[8][4];
        #   load_q_global(q_reg_fl, ...);
        # In FlyDSL q_reg_fl[8][4] float2 -> [8] of v8f32 (8 packed f32 per j).
        q_reg_fl = [None] * 8
        q_row_offset = lid % 32
        q_col_offset = 8 * (lid // 32)
        q_base = (
            batch_idx * (ATTN_N * ATTN_H * ATTN_D)
            + tile_idx * (Q_BLOCK_SIZE * ATTN_H * ATTN_D)
            + head_idx * ATTN_D
        )
        q_sc8 = Vec.from_elements([f32(TEMPERATURE_SCALE)], f32).broadcast_to(8).ir_value()
        for j in range_constexpr(8):
            col = 16 * j + q_col_offset
            elem_off = q_base + q_row_offset * Q_stride1 + col
            raw = buffer_ops.buffer_load(Q_rsrc, _raw(i32(elem_off)), vec_width=8, dtype=bf16)
            vf = Vec(raw).to(f32)
            scaled = arith.mulf(_raw(vf), _raw(q_sc8), fastmath=fm)
            q_reg_fl[j] = scaled
            # convert_q_to_bf16: f32 pack -> bf16 pack
            qbf = Vec(scaled).to(bf16).ir_value()
            q_reg[0][j] = qbf
            # transpose_q (q_reg[0][j] -> q_reg_t[j][0]): identity relabel
            q_reg_t[j][0] = qbf

        # ---------------- Load K[1] into shared, V[0] into shared ----------------
        # Mirrors hip_kernel.cpp:
        #   group_load_srsrc<true>(1, ...);   // K[1] -> k_smem_1
        #   group_load_srsrc<false>(0, ...);  // V[0] -> v_smem_0
        load_k(1, 1)
        load_v(0, 0)

        # ---------------- Load K[0] from shared to registers ----------------
        # Mirrors hip_kernel.cpp:
        #   load_k_from_shared(k_reg, k_smem_0);
        #   __builtin_amdgcn_sched_barrier(0);
        #   s_waitcnt lgkmcnt(0);
        #   s_waitcnt vmcnt(2);
        #   __builtin_amdgcn_sched_barrier(0);
        #   __builtin_amdgcn_s_barrier();
        k_reg = load_k_regs(0)
        rocdl.sched_barrier(0)
        wait_lgkmcnt0()
        wait_vmcnt(2)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # ---------------- QK[0] ----------------
        # Mirrors hip_kernel.cpp:
        #   zero_att(att_block[0]);
        #   transpose_k(k_reg_t, k_reg);
        #   mma_AtB_QK(att_block[0], k_reg_t, q_reg_t, att_block[0]);
        # QK accumulates into the float2 tile att_block[0] (att_block_bf16 is
        # filled later by copy_f32_to_bf16). zero_att seeds both heights to 0.
        att_block[0] = [ZERO16, ZERO16]

        # transpose_k: k_reg_t[j][i] = k_reg[i][j] (identity relabel of v8bf16)
        for i in range_constexpr(2):
            for j in range_constexpr(8):
                k_reg_t[j][i] = k_reg[i][j]

        # mma_AtB_QK: att_block[0][n] = sum_k k_reg_t[k][n] * q_reg_t[k][0]
        att_block[0] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[0])

        # ---------------- Partial softmax for QK[0] ----------------
        # Mirrors hip_kernel.cpp:
        #   max_vec = col_max_reset(att_block[0]);
        #   max_vec_prev = max_vec;
        #   sub_col_att(att_block[0], max_vec);
        #   exp2_base(att_block[0][0]);            // height 0 only
        #   if (stagger) { sched_barrier(0); s_barrier(); }
        max_vec = fx.Float32(col_max(att_block[0]))
        max_vec_prev = max_vec
        att_block[0] = sub_col(att_block[0], max_vec)
        att_block[0][0] = exp2_one(att_block[0][0])

        rocdl.sched_barrier(0)

        if stagger > 0:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

        rocdl.sched_barrier(0)

        # ---- Load K[1] from shared, load K[2] into shared, load V[1] into shared ----
        # Mirrors hip_kernel.cpp:
        #   load_k_from_shared(k_reg, k_smem_1);
        #   group_load_srsrc<true>(2, ...);   // K[2] -> k_smem_0
        #   group_load_srsrc<false>(1, ...);  // V[1] -> v_smem_1
        #   s_waitcnt lgkmcnt(0);
        #   s_waitcnt vmcnt(4);
        #   __builtin_amdgcn_sched_barrier(0);
        #   __builtin_amdgcn_s_barrier();
        k_reg = load_k_regs(1)
        load_k(2, 0)
        load_v(1, 1)
        wait_lgkmcnt0()
        wait_vmcnt(4)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # pending_scale: i1, deferred norm rescale flag (reference's int pending_scale).
        pending_scale = arith.cmpi(arith.CmpIPredicate.ne, _raw(i32(0)), _raw(i32(0)))

        # ---- hot-loop helpers ----
        # copy_f32_to_bf16_half: one height (v16f32) -> 2 OV packs (v8bf16).
        def copy_half(att_h):
            bv = Vec(att_h).to(bf16)
            return [
                bv.shuffle(bv, list(range(0, 8))).ir_value(),
                bv.shuffle(bv, list(range(8, 16))).ir_value(),
            ]

        # Deferred norm rescale: norm = pending ? norm*scale : norm.
        def apply_pending_norm(norm, pending, scale):
            scaled = arith.mulf(_raw(norm), _raw(scale), fastmath=fm)
            return arith.select(_raw(pending), _raw(scaled), _raw(norm))

        # Lazy-threshold rescale with deferred norm (mirrors reference cluster 2/6):
        #   max_vec = col_max_accum(att, max_prev); lane vote on (cur-prev) > T.
        #   all_ok  -> keep max_prev, o unchanged, pending=0.
        #   else    -> max_prev=m_new, o *= exp2(prev-new), pending=1.
        # Returns (o_new, scale_s, pending_i1, kept_max). max_vec == max_vec_prev
        # == kept_max after the block in both branches, so callers set both.
        def rescale_defer(att_buf, o_reg, max_prev, scale_old):
            m_cur = col_max(att_buf)
            m_new = fmaxf(m_cur, max_prev)
            delta = arith.subf(_raw(m_new), _raw(max_prev), fastmath=fm)
            not_ok = arith.cmpf(
                arith.CmpFPredicate.OGT, _raw(delta), _raw(f32(RESCALE_THRESHOLD))
            )
            mask = rocdl.ballot(T.i64, not_ok)
            pending = arith.cmpi(arith.CmpIPredicate.ne, _raw(mask), _raw(fx.Int64(0)))
            scale_exp = Vec(
                bcast16(arith.subf(_raw(max_prev), _raw(m_new), fastmath=fm))
            ).exp2().ir_value()
            scale_s = Vec(scale_exp)[0]
            # o-multiply factor: scale on rescale, else 1.0 (o unchanged).
            corr_v = bcast16(arith.select(pending, _raw(scale_s), _raw(f32(1.0))))
            # scale_vec is only updated on rescale; otherwise carried forward, so
            # the epilogue's unconditional norm *= scale_vec matches the reference.
            scale_new = fx.Float32(arith.select(pending, _raw(scale_s), _raw(scale_old)))
            kept_max = arith.select(pending, _raw(m_new), _raw(max_prev))
            o_new = [
                arith.mulf(_raw(o_reg[n]), _raw(corr_v), fastmath=fm)
                for n in range_constexpr(4)
            ]
            return o_new, scale_new, pending, fx.Float32(kept_max)

        # ========================================================================
        # Hot loop (mirrors hip_kernel.cpp clusters 0-7, j = 3,5,...,num_tiles-2).
        # Body factored into hot_iter (pure: takes carried state, returns new state,
        # mutates no outer vars) so it can be driven by a rolled runtime loop. A
        # rolled loop keeps o_reg/k_reg as loop-carried phis (fixed registers across
        # the back-edge) instead of one giant unrolled live range that spills.
        # Buffer parities: odd tiles use buf1 for K-shared/V-store, even use buf0.
        # ========================================================================
        def hot_iter(j, k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale):
            jm1 = j - 1
            jp1 = j + 1
            k_reg_t = [[None] * 2 for _ in range_constexpr(8)]

            # ---- Cluster 0: QK[odd] + finish softmax for QK[even] ----
            att1 = [ZERO16, ZERO16]
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    k_reg_t[jj][i] = k_reg[i][jj]

            norm_vec = apply_pending_norm(norm_vec, pending_scale, scale_vec)

            att1 = mma_AtB_QK(k_reg_t, q_reg_t, att1)
            even_lo = copy_half(att0[0])                  # even height0 -> packs 0,1
            att0[1] = exp2_one(att0[1])                   # finish even height1
            norm_vec = col_sum_into(att0, norm_vec)
            even_hi = copy_half(att0[1])                  # even height1 -> packs 2,3
            att_block_bf16 = [even_lo[0], even_lo[1], even_hi[0], even_hi[1]]
            sched_exp_pairs(6, 3, 1)
            sched_pairs(10, 5, 1)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 1: Load K[j] into shared (buf1), load V from shared (buf0) ----
            load_k(j, 1)
            v_reg = load_v_regs(0)
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 2: A[even]*V, partial softmax for QK[odd] ----
            rocdl.s_setprio(1)
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[0], att_block_bf16[0])
            o_reg, scale_vec, pending_scale, max_vec = rescale_defer(
                att1, o_reg, max_vec_prev, scale_vec)
            max_vec_prev = max_vec
            sched_pairs(4, 5, 2)

            o_reg = mma_AtB_OV_slice(o_reg, v_reg[1], att_block_bf16[1])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[2], att_block_bf16[2])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[3], att_block_bf16[3])
            att1 = sub_col(att1, max_vec)
            att1[0] = exp2_one(att1[0])
            sched_pairs(6, 5, 2)
            sched_exp_pairs(6, 3, 2)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 3: Load V[j-1] into shared (buf0), load K from shared (buf0) ----
            load_v(jm1, 0)
            k_reg = load_k_regs(0)
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 4: QK[even] + finish softmax for QK[odd] ----
            att0 = [ZERO16, ZERO16]
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    k_reg_t[jj][i] = k_reg[i][jj]

            norm_vec = apply_pending_norm(norm_vec, pending_scale, scale_vec)

            att0 = mma_AtB_QK(k_reg_t, q_reg_t, att0)
            att1[1] = exp2_one(att1[1])                   # finish odd height1
            norm_vec = col_sum_into(att1, norm_vec)
            att_block_bf16 = to_bf16_packs(att1)          # full copy of odd
            sched_exp_pairs(6, 3, 3)
            sched_pairs(10, 5, 3)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 5: Load K[j+1] into shared (buf0), load V from shared (buf1) ----
            load_k(jp1, 0)
            v_reg = load_v_regs(1)
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 6: A[odd]*V, partial softmax for QK[even] ----
            rocdl.s_setprio(1)
            mma_AtB_OV_slice(o_reg, v_reg[0], att_block_bf16[0])
            o_reg, scale_vec, pending_scale, max_vec = rescale_defer(att0, o_reg, max_vec_prev, scale_vec)
            max_vec_prev = max_vec
            sched_pairs(4, 5, 4)
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[1], att_block_bf16[1])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[2], att_block_bf16[2])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[3], att_block_bf16[3])
            att0 = sub_col(att0, max_vec)
            att0[0] = exp2_one(att0[0])
            sched_pairs(6, 5, 4)
            sched_exp_pairs(6, 3, 4)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Cluster 7: Load V[j] into shared (buf1), load K from shared (buf1) ----
            load_v(j, 1)
            k_reg = load_k_regs(1)
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            return k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale

        # Flatten/unflatten the carried state to/from a flat list of raw ir.Values,
        # as required by the runtime range(..., init=) loop-carried phi mechanism.
        #   layout: k_reg[2][8] (16) | att0[2] (2) | o_reg[4] (4) | max_vec_prev,
        #           norm_vec, scale_vec (3 f32) | pending_scale (1 i1)  = 26 values
        def _flatten(k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale):
            flat = []
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    flat.append(_raw(k_reg[i][jj]))
            flat.append(_raw(att0[0])); flat.append(_raw(att0[1]))
            for n in range_constexpr(4):
                flat.append(_raw(o_reg[n]))
            flat.append(_raw(max_vec_prev)); flat.append(_raw(norm_vec))
            flat.append(_raw(scale_vec)); flat.append(_raw(pending_scale))
            return flat

        def _unflatten(flat):
            p = 0
            k_reg = [[None] * 8 for _ in range_constexpr(2)]
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    k_reg[i][jj] = flat[p]; p += 1
            att0 = [flat[p], flat[p + 1]]; p += 2
            o_reg = [flat[p + n] for n in range_constexpr(4)]; p += 4
            max_vec_prev = fx.Float32(flat[p]); p += 1
            norm_vec = fx.Float32(flat[p]); p += 1
            scale_vec = fx.Float32(flat[p]); p += 1
            pending_scale = flat[p]; p += 1
            return k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale

        init_flat = _flatten(k_reg, att_block[0], o_reg, max_vec_prev,
                             norm_vec, scale_vec, pending_scale)
        _lo = fx.Index(3)
        _hi = fx.Index(NUM_KV_TILES - 1)
        _step = fx.Index(2)
        loop_results = init_flat
        for j, carried in range(_lo, _hi, _step, init=init_flat):
            j_i32 = i32(j)
            st = _unflatten(carried)
            st = hot_iter(j_i32, *st)
            loop_results = yield _flatten(*st)
        (k_reg, att_block[0], o_reg, max_vec_prev, norm_vec, scale_vec,
         pending_scale) = _unflatten(loop_results)

        # ====================================================================
        # Epilogue (mirrors hip_kernel.cpp clusters 0-12). Unconditional rescale
        # (no lazy-threshold vote): every tile rescales o_reg/norm by exp2(prev-new).
        # ====================================================================
        # full OV: o_reg[n] += sum_kk v_reg[kk][n] * att_bf[kk] (4 contraction slices)
        def full_ov(o_reg, vreg, packs):
            for kk in range_constexpr(4):
                o_reg = ov_slice(o_reg, [vreg[kk][n] for n in range_constexpr(4)], packs[kk])
            return o_reg

        # Unconditional rescale (scale only, no o-multiply): returns
        #   (scale_s, max_new). max_new = max(col_max(att), max_prev);
        #   scale = exp2(max_prev - max_new). The reference emits mul_col_o late
        #   (after the sched barriers), so callers apply mul_o explicitly.
        def rescale_uncond(att_buf, max_prev):
            m_new = fmaxf(col_max(att_buf), max_prev)
            scale_s = Vec(
                bcast16(arith.subf(_raw(max_prev), _raw(m_new), fastmath=fm))
            ).exp2().ir_value()
            return fx.Float32(Vec(scale_s)[0]), fx.Float32(m_new)

        nt = NUM_KV_TILES

        # ---- Cluster 0: QK[last odd] + finish softmax for last even ----
        att_block[1] = [ZERO16, ZERO16]
        for i in range_constexpr(2):
            for jj in range_constexpr(8):
                k_reg_t[jj][i] = k_reg[i][jj]
        att_block[1] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[1])
        att_block[0][1] = exp2_one(att_block[0][1])
        norm_vec = fmul(norm_vec, scale_vec)
        norm_vec = col_sum_into(att_block[0], norm_vec)
        att_block_bf16 = to_bf16_packs(att_block[0])
        sched_exp_pairs(6, 3, 5)
        sched_pairs(10, 5, 5)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 1: Load K[nt-1] into shared (buf1), load V from shared (buf0) ----
        load_k(nt - 1, 1)
        v_reg = load_v_regs(0)
        wait_lgkmcnt0()
        wait_vmcnt(4)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 2: A*V, partial softmax for last odd ----
        rocdl.s_setprio(1)
        o_reg = full_ov(o_reg, v_reg, att_block_bf16)
        scale_vec, max_vec = rescale_uncond(att_block[1], max_vec_prev)
        max_vec_prev = max_vec
        att_block[1] = sub_col(att_block[1], max_vec)
        att_block[1][0] = exp2_one(att_block[1][0])
        sched_pairs(10, 5, 6)
        sched_exp_pairs(6, 3, 6)
        rocdl.sched_barrier(0)
        o_reg = mul_o(o_reg, scale_vec)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 3: Load V[nt-2] into shared (buf0), load K from shared (buf0) ----
        load_v(nt - 2, 0)
        k_reg = load_k_regs(0)
        wait_lgkmcnt0()
        wait_vmcnt(4)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 4: QK + finish softmax for the odd from cluster 2 ----
        att_block[0] = [ZERO16, ZERO16]
        for i in range_constexpr(2):
            for jj in range_constexpr(8):
                k_reg_t[jj][i] = k_reg[i][jj]
        att_block[0] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[0])
        att_block[1][1] = exp2_one(att_block[1][1])
        norm_vec = fmul(norm_vec, scale_vec)
        norm_vec = col_sum_into(att_block[1], norm_vec)
        att_block_bf16 = to_bf16_packs(att_block[1])
        sched_exp_pairs(6, 3, 7)
        sched_pairs(10, 5, 7)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 5: Load V from shared (buf1) ----
        v_reg = load_v_regs(1)
        wait_lgkmcnt0()
        wait_vmcnt(2)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 6: A*V, partial softmax for the even from cluster 4 ----
        rocdl.s_setprio(1)
        o_reg = full_ov(o_reg, v_reg, att_block_bf16)
        scale_vec, max_vec = rescale_uncond(att_block[0], max_vec_prev)
        max_vec_prev = max_vec
        att_block[0] = sub_col(att_block[0], max_vec)
        att_block[0][0] = exp2_one(att_block[0][0])
        sched_pairs(10, 5, 8)
        sched_exp_pairs(6, 3, 8)
        rocdl.sched_barrier(0)
        o_reg = mul_o(o_reg, scale_vec)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 7: Load V[nt-1] into shared (buf1), load K from shared (buf1) ----
        load_v(nt - 1, 1)
        k_reg = load_k_regs(1)
        wait_lgkmcnt0()
        wait_vmcnt(2)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 8: QK + finish softmax for the even from cluster 6 ----
        att_block[1] = [ZERO16, ZERO16]
        for i in range_constexpr(2):
            for jj in range_constexpr(8):
                k_reg_t[jj][i] = k_reg[i][jj]
        att_block[1] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[1])
        att_block[0][1] = exp2_one(att_block[0][1])
        norm_vec = fmul(norm_vec, scale_vec)
        norm_vec = col_sum_into(att_block[0], norm_vec)
        att_block_bf16 = to_bf16_packs(att_block[0])
        sched_exp_pairs(6, 3, 9)
        sched_pairs(10, 5, 9)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 9: Load V from shared (buf0) ----
        v_reg = load_v_regs(0)
        wait_lgkmcnt0()
        wait_vmcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 10: A*V, full softmax for the last QK (att_block[1]) ----
        o_reg = mma_AtB_OV(o_reg, v_reg, att_block_bf16)
        scale_vec, max_vec = rescale_uncond(att_block[1], max_vec_prev)
        max_vec_prev = max_vec
        att_block[1] = sub_col(att_block[1], max_vec)
        att_block[1][0] = exp2_one(att_block[1][0])
        sched_pairs(10, 5, 10)
        sched_exp_pairs(6, 3, 10)
        rocdl.sched_barrier(0)
        att_block[1][1] = exp2_one(att_block[1][1])
        norm_vec = fmul(norm_vec, scale_vec)
        norm_vec = col_sum_into(att_block[1], norm_vec)
        att_block_bf16 = to_bf16_packs(att_block[1])
        rocdl.sched_barrier(0)
        o_reg = mul_o(o_reg, scale_vec)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 11: Load V from shared (buf1) ----
        v_reg = load_v_regs(1)
        wait_lgkmcnt0()
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Cluster 12: Final A*V and normalize ----
        o_reg = mma_AtB_OV(o_reg, v_reg, att_block_bf16)
        # Single reciprocal + broadcast multiply (mirrors HIP div_col_o). Using an
        # opaque rocdl.rcp instead of fast-math divf stops the compiler from
        # re-expanding this into a full IEEE division per O element (64 divisions).
        inv = rocdl.rcp(T.f32, _raw(norm_vec))
        o_reg = mul_o(o_reg, inv)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Conclusion: store O and LSE ----
        not_stagger_cond = arith.cmpi(
            arith.CmpIPredicate.eq, _raw(i32(stagger)), _raw(i32(0))
        )
        ns_if = _scf.IfOp(not_stagger_cond, [], has_else=False, loc=ir.Location.unknown())
        if len(ns_if.regions[0].blocks) == 0:
            ns_if.regions[0].blocks.append(*[])
        with ir.InsertionPoint(ns_if.regions[0].blocks[0]):
            rocdl.s_barrier()
            _scf.YieldOp([])

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
