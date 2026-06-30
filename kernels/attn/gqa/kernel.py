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
        # lds_base is a wave-uniform SGPR byte address (per-warp LDS destination),
        # hoisted once via readfirstlane in the setup below — mirrors the reference's
        # k_lds_base_0/1 and v_lds_base_0/1 so per-buffer LDS addresses don't stay
        # live in VGPRs across the kernel.
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
                    vbf = Vec.from_elements(elems, f32).to(bf16)
                    for e in range_constexpr(4):
                        off = base + row_offset * O_stride1 + col + e
                        buffer_ops.buffer_store(vbf[e], O_rsrc, _raw(i32(off)))

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

        def wait_vmcnt(n):
            _llvm.inline_asm(None, [], f"s_waitcnt vmcnt({n})", "", has_side_effects=True)

        def wait_lgkmcnt0():
            _llvm.inline_asm(None, [], "s_waitcnt lgkmcnt(0)", "", has_side_effects=True)

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

        # =====================================================================
        # Imperative setup (all helpers above; executable code below)
        # =====================================================================
        ZERO16 = Vec.filled(16, 0.0, f32).ir_value()

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

        # Per-warp LDS destination bases, hoisted into SGPRs (wave-uniform) so the
        # global->LDS loads don't keep per-buffer VGPR addresses live across the
        # kernel. Mirrors hip_kernel.cpp's k_lds_base_0/1, v_lds_base_0/1.
        lds_warp_off = wid * BYTES_PER_WARP
        def _lds_base(smem_ptr):
            byte_addr = i32(fx.ptrtoint(smem_ptr)) + i32(lds_warp_off)
            return i32(arith.unwrap(rocdl.readfirstlane(T.i32, _raw(byte_addr))))
        k_lds_base_0 = _lds_base(k_smem_0)
        k_lds_base_1 = _lds_base(k_smem_1)
        v_lds_base_0 = _lds_base(v_smem_0)
        v_lds_base_1 = _lds_base(v_smem_1)
        k_lds_base = [k_lds_base_0, k_lds_base_1]
        v_lds_base = [v_lds_base_0, v_lds_base_1]

        bx = fx.block_idx.x
        head_idx = (bx % ATTN_H_KV) * GROUP_SIZE + (bx // ATTN_H_KV)
        batch_idx = fx.block_idx.z
        head_idx_kv = head_idx // GROUP_SIZE
        block_tile_idx = fx.block_idx.y
        tile_idx = i32(arith.unwrap(rocdl.readfirstlane(T.i32, _raw(block_tile_idx * NUM_WARPS + wid))))
        stagger = wid // 4

        k_base_elems = batch_idx * (ATTN_N * ATTN_H_KV * ATTN_D) + head_idx_kv * ATTN_D
        v_base_elems = batch_idx * (ATTN_N * ATTN_H_KV * ATTN_D) + head_idx_kv * ATTN_D

        # Single base descriptor per tensor (mirrors k_srsrc_base / v_srsrc_base in
        # hip_kernel.cpp); both double-buffer slots share the same resource.
        k_srsrc_base = buffer_ops.create_buffer_resource(K)
        v_srsrc_base = buffer_ops.create_buffer_resource(V)
        Q_rsrc = buffer_ops.create_buffer_resource(Q)
        k_rsrc = [k_srsrc_base, k_srsrc_base]
        v_rsrc = [v_srsrc_base, v_srsrc_base]

        off_K = prefill_offsets(True, K_stride1)
        off_V = prefill_offsets(False, V_stride1)

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

        # One pipelined KV tile: QK[t] overlaps finish/OV[t-1]. Identical body to
        # the original per-tile step; pulled into a helper so the hot loop can be a
        # runtime scf.for (step 2) with compile-time-constant double-buffer parities
        # (cur/nxt), which keeps register pressure bounded instead of fully
        # unrolling all KV tiles (the latter spills ~600 VGPRs to scratch).
        def tile_step(cur, nxt, prefetch_tile, has_next, att_held, o_reg,
                      vreg_held, max_vec, norm_vec, wait_n):
            kreg = load_k_regs(cur)

            # QK[t] (frontier) overlaps with finishing tile t-1
            rocdl.s_setprio(1)
            att_next = qk(kreg, qpacks)

            # finish tile t-1: exp height1, col_sum, OV
            o_reg, norm_vec = stage_finish(att_held, vreg_held, o_reg, norm_vec)

            # softmax start for tile t (rescale must follow tile t-1's OV)
            corr_v, corr_s, max_vec = lazy_rescale(att_next, max_vec)
            norm_vec = fmul(norm_vec, corr_s)
            o_reg = [arith.mulf(_raw(o_reg[n]), _raw(corr_v), fastmath=fm) for n in range_constexpr(4)]
            sub = sub_col(att_next, max_vec)
            sub[0] = exp2_one(sub[0])
            sched_exp_pairs(6, 3, 1)
            sched_pairs(10, 5, 1)
            rocdl.s_setprio(0)
            sb0()
            bar()
            sb0()

            # prefetch tile t+1 into the alternate buffer, then load tile t's V regs
            if const_expr(has_next):
                load_k(prefetch_tile, nxt)
                load_v(prefetch_tile, nxt)
            vreg_new = load_v_regs(cur)
            wait_lgkmcnt0()
            wait_vmcnt(wait_n)
            sb0()
            bar()
            sb0()
            return sub, o_reg, vreg_new, max_vec, norm_vec

        # ---- prologue: stage K[0]/V[0] into buf0, K[1]/V[1] into buf1 ----
        load_k(0, 0)
        load_v(0, 0)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

        kreg = load_k_regs(0)
        if const_expr(NUM_KV_TILES > 1):
            load_k(1, 1)
            load_v(1, 1)
        rocdl.s_setprio(1)
        att_held, max_vec, o_reg, norm_vec = stage_qk(kreg, max_vec, o_reg, norm_vec, True)
        rocdl.s_setprio(0)
        sb0()
        bar()
        sb0()
        vreg_held = load_v_regs(0)
        wait_lgkmcnt0()
        wait_vmcnt(2)
        sb0()
        bar()
        sb0()

        # ---- flatten loop-carried state into scalar iter_args (scf.for requires
        #      each carried value to be a single MLIR value, not a Python list) ----
        ah0, ah1 = att_held[0], att_held[1]
        o0, o1, o2, o3 = o_reg[0], o_reg[1], o_reg[2], o_reg[3]
        vh0, vh1, vh2, vh3 = vreg_held[0][0], vreg_held[0][1], vreg_held[0][2], vreg_held[0][3]
        vh4, vh5, vh6, vh7 = vreg_held[1][0], vreg_held[1][1], vreg_held[1][2], vreg_held[1][3]
        vh8, vh9, vh10, vh11 = vreg_held[2][0], vreg_held[2][1], vreg_held[2][2], vreg_held[2][3]
        vh12, vh13, vh14, vh15 = vreg_held[3][0], vreg_held[3][1], vreg_held[3][2], vreg_held[3][3]

        # ---- hot loop: runtime scf.for over tile pairs. jj is always odd so the
        #      two tiles per body have fixed parities (cur=1 then cur=0). Covers
        #      tiles 1..NUM_KV_TILES-2; the last tile is handled below. ----
        # Only flat scalar/vector names (ah*, o*, vh*, max_vec, norm_vec) are
        # loop-carried; the reconstructed Python lists below use fresh names so the
        # auto-iter_arg collector doesn't try to carry a list (which scf.for rejects).
        for jj in range(1, NUM_KV_TILES - 2, 2):
            jt = i32(jj)
            _ah = [ah0, ah1]
            _o = [o0, o1, o2, o3]
            _vh = [[vh0, vh1, vh2, vh3], [vh4, vh5, vh6, vh7],
                   [vh8, vh9, vh10, vh11], [vh12, vh13, vh14, vh15]]

            # tile t = jj (parity 1, prefetch jj+1 into buf 0)
            _ah, _o, _vh, max_vec, norm_vec = tile_step(
                1, 0, jt + 1, True, _ah, _o, _vh, max_vec, norm_vec, 4)
            # tile t = jj+1 (parity 0, prefetch jj+2 into buf 1)
            _ah, _o, _vh, max_vec, norm_vec = tile_step(
                0, 1, jt + 2, True, _ah, _o, _vh, max_vec, norm_vec, 4)

            ah0, ah1 = _ah[0], _ah[1]
            o0, o1, o2, o3 = _o[0], _o[1], _o[2], _o[3]
            vh0, vh1, vh2, vh3 = _vh[0][0], _vh[0][1], _vh[0][2], _vh[0][3]
            vh4, vh5, vh6, vh7 = _vh[1][0], _vh[1][1], _vh[1][2], _vh[1][3]
            vh8, vh9, vh10, vh11 = _vh[2][0], _vh[2][1], _vh[2][2], _vh[2][3]
            vh12, vh13, vh14, vh15 = _vh[3][0], _vh[3][1], _vh[3][2], _vh[3][3]

        # ---- leftover final tile t = NUM_KV_TILES-1 (parity 1, no prefetch) ----
        att_held = [ah0, ah1]
        o_reg = [o0, o1, o2, o3]
        vreg_held = [[vh0, vh1, vh2, vh3], [vh4, vh5, vh6, vh7],
                     [vh8, vh9, vh10, vh11], [vh12, vh13, vh14, vh15]]
        att_held, o_reg, vreg_held, max_vec, norm_vec = tile_step(
            1, 0, 0, False, att_held, o_reg, vreg_held, max_vec, norm_vec, 4)

        # ---- epilogue: finish the last held tile ----
        o_reg, norm_vec = stage_finish(att_held, vreg_held, o_reg, norm_vec)

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
