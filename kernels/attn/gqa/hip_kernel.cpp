#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <bit>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// ============================================================================
// Constants
// ============================================================================
#ifndef ATTN_B
constexpr int ATTN_B = 16;
#endif
#ifndef ATTN_H
constexpr int ATTN_H = 64;
#endif
#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8;
#endif
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
#ifndef ATTN_N
constexpr int ATTN_N = 1024;
#endif
constexpr int ATTN_D = 128;
constexpr int Q_BLOCK_SIZE = 32;
constexpr int KV_BLOCK_SIZE = 64;
// Number of 32-wide query column-tiles in the attention block.
// This is the per-lane length of the softmax row-vectors (max/norm/scale),
// mirroring attn_tile::row_vec::outer_dim in the reference kernel.cpp.
constexpr int ATT_WIDTH = Q_BLOCK_SIZE / 32;
#define NUM_WARPS 8
constexpr int WARP_SIZE = 64;
#define NUM_THREADS (WARP_SIZE * NUM_WARPS)

constexpr int NUM_KV_TILES = ATTN_N / KV_BLOCK_SIZE;

// ============================================================================
// Type aliases
// ============================================================================
using bf16   = __hip_bfloat16;
using bf16_2 = __hip_bfloat162;

using i32x4      = int32_t __attribute__((ext_vector_type(4)));
using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

typedef uint32_t      uint2_t __attribute__((ext_vector_type(2)));

typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

// ============================================================================
// Intrinsic declarations
// ============================================================================
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, 0x110000);
    row_stride_bytes &= 0x3FFF;
    if (row_stride_bytes) {
        uint64_t stride_field = row_stride_bytes;
        stride_field = stride_field | 0x4000;
        stride_field = stride_field | 0x8000;
        rsrc.ptr |= stride_field << 48;
    }
    return *reinterpret_cast<const i32x4*>(&rsrc);
}

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");

extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                int size,
                                int voffset,
                                int soffset,
                                int offset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

// ============================================================================
// Device helpers
// ============================================================================
__device__ inline int laneid() {
    return threadIdx.x % WARP_SIZE;
}
__device__ inline int warpid() {
    return threadIdx.x / WARP_SIZE;
}

// ============================================================================
// Shared memory swizzle
// ============================================================================

// st_32x32 swizzle (for K tiles)
__device__ __forceinline__ uint32_t swizzle_k(int row, int col) {
    uint32_t offset = 2u * (row * 32 + col);
    uint32_t s1 = ((offset % 1024u) >> 9) << 5;
    uint32_t s2 = ((offset % 2048u) >> 10) << 4;
    return offset ^ s1 ^ s2;
}

// st_8x32 swizzle (for V tiles) — identity
__device__ __forceinline__ uint32_t swizzle_v(int row, int col) {
    return 2u * (row * 32 + col);
}

// ============================================================================
// MFMA wrapper
// ============================================================================
__device__ __forceinline__ void mfma_f32_32x32x16_bf16(
    float2 (&D)[8], const bf16_2 (&A)[4], const bf16_2 (&B)[4], const float2 (&C)[8])
{
    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(const bf16x8_t*)A, *(const bf16x8_t*)B, *(const floatx16_t*)C, 0, 0, 0);
}

// ============================================================================
// Scheduler barrier templates
// ============================================================================
#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define EXP_MASK  0x400

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

template<int Pairs, int EXP_CNT, int Group>
__device__ __forceinline__ void sched_barrier_exp_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}

// ============================================================================
// mma_AtB: QK multiply
//   att_block[2][8] = k_t[8][2][4]^T × q_t[8][4]
//   D(height=2,width=1), A(height=8,width=2), B(height=8,width=1)
// ============================================================================
__device__ __forceinline__ void mma_AtB_QK(
    float2 (&D)[2][ATT_WIDTH][8],
    const bf16_2 (&A)[8][2][4],
    const bf16_2 (&B)[ATT_WIDTH][8][4],
    const float2 (&C)[2][ATT_WIDTH][8])
{
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            mfma_f32_32x32x16_bf16(D[n][w], A[0][n], B[w][0], C[n][w]);
            #pragma unroll
            for (int k = 1; k < 8; k++) {
                mfma_f32_32x32x16_bf16(D[n][w], A[k][n], B[w][k], D[n][w]);
            }
        }
    }
}

// ============================================================================
// mma_AtB: OV multiply
//   o_reg[4][8] += v[4][4][4]^T × att_in[4][4]
//   D(height=4,width=1), A(height=4,width=4), B(height=4,width=1)
// ============================================================================
__device__ __forceinline__ void mma_AtB_OV(
    float2 (&D)[4][ATT_WIDTH][8],
    const bf16_2 (&A)[4][4][4],
    const bf16_2 (&B)[ATT_WIDTH][4][4],
    const float2 (&C)[4][ATT_WIDTH][8])
{
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            mfma_f32_32x32x16_bf16(D[n][w], A[0][n], B[w][0], C[n][w]);
            #pragma unroll
            for (int k = 1; k < 4; k++) {
                mfma_f32_32x32x16_bf16(D[n][w], A[k][n], B[w][k], D[n][w]);
            }
        }
    }
}

// ============================================================================
// Transpose: k_reg[2][8][4] -> k_reg_t[8][2][4]
// ============================================================================
__device__ __forceinline__ void transpose_k(
    bf16_2 (&dst)[8][2][4], const bf16_2 (&src)[2][8][4])
{
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++)
                __builtin_memcpy(&dst[j][i][k], &src[i][j][k], sizeof(bf16_2));
}

// ============================================================================
// Copy float2 -> bf16_2 (att_block -> att_block_bf16)
// ============================================================================
__device__ __forceinline__ void copy_f32_to_bf16(
    bf16_2 (&dst)[2][ATT_WIDTH][8], const float2 (&src)[2][ATT_WIDTH][8])
{
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < ATT_WIDTH; j++)
            #pragma unroll
            for (int k = 0; k < 8; k++)
                dst[i][j][k] = __float22bfloat162_rn(src[i][j][k]);
}

// ============================================================================
// Reinterpret rt_32x32 bf16 [2][8] -> rt_16x32_4 bf16 [4][4]
// dst[i][k] = src[i/2][(i%2)*4 + k]
// ============================================================================
__device__ __forceinline__ void reinterpret_att(
    bf16_2 (&dst)[ATT_WIDTH][4][4], const bf16_2 (&src)[2][ATT_WIDTH][8])
{
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++)
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int k = 0; k < 4; k++)
                dst[j][i][k] = src[i / 2][j][(i % 2) * 4 + k];
}

// ============================================================================
// Rescale threshold (skip rescaling when max hasn't grown significantly)
// ============================================================================
constexpr float RESCALE_THRESHOLD = 8.0f;

// ============================================================================
// Row-vector (per-width) helpers for the softmax accumulators
// ============================================================================
__device__ __forceinline__ void rv_copy(float (&dst)[ATT_WIDTH], const float (&src)[ATT_WIDTH]) {
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) dst[w] = src[w];
}
__device__ __forceinline__ void rv_exp2(float (&v)[ATT_WIDTH]) {
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) v[w] = exp2f(v[w]);
}
__device__ __forceinline__ void rv_mul(float (&dst)[ATT_WIDTH], const float (&src)[ATT_WIDTH]) {
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) dst[w] *= src[w];
}
// scale_vec = exp2(prev - cur), per width
__device__ __forceinline__ void rv_diff_exp2(
    float (&dst)[ATT_WIDTH], const float (&prev)[ATT_WIDTH], const float (&cur)[ATT_WIDTH]) {
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) dst[w] = exp2f(prev[w] - cur[w]);
}
// true iff EVERY width slot is within threshold (wave-uniform vote)
__device__ __forceinline__ int rv_all_below(
    const float (&prev)[ATT_WIDTH], const float (&cur)[ATT_WIDTH], float thresh) {
    int ok = 1;
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) ok &= (cur[w] - prev[w] <= thresh);
    return __all(ok);
}

// ============================================================================
// Softmax helpers
// ============================================================================

// col_max: per-width column reduction of the att tile to its row-vector.
// For each of the ATT_WIDTH query column-tiles, reduce the in-lane elements
// (height x packed) then merge the two 32-row halves with permlane32_swap.
// Mirrors col_reduce<max> over a col_l rt_32x32 tile in the reference.
__device__ __forceinline__ void col_max_reset(
    float (&mx)[ATT_WIDTH], const float2 (&data)[2][ATT_WIDTH][8]) {
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++) {
        float m = -__builtin_huge_valf();
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                m = __builtin_fmaxf(m, data[i][j][k].x);
                m = __builtin_fmaxf(m, data[i][j][k].y);
            }
        uint2_t res = __builtin_amdgcn_permlane32_swap(
            __float_as_uint(m), __float_as_uint(m), false, true);
        mx[j] = __builtin_fmaxf(__uint_as_float(res.x), __uint_as_float(res.y));
    }
}

// col_max accumulating onto a previous row-vector (per width).
__device__ __forceinline__ void col_max_accum(
    float (&mx)[ATT_WIDTH], const float2 (&data)[2][ATT_WIDTH][8],
    const float (&prev)[ATT_WIDTH]) {
    col_max_reset(mx, data);
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++)
        mx[j] = __builtin_fmaxf(mx[j], prev[j]);
}

// col_sum: per-width column reduction, accumulating onto prev row-vector.
__device__ __forceinline__ void col_sum_accum(
    float (&sm_out)[ATT_WIDTH], const float2 (&data)[2][ATT_WIDTH][8],
    const float (&prev)[ATT_WIDTH]) {
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++) {
        float sm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sm += data[i][j][k].x;
                sm += data[i][j][k].y;
            }
        uint2_t res = __builtin_amdgcn_permlane32_swap(
            __float_as_uint(sm), __float_as_uint(sm), false, true);
        sm = __uint_as_float(res.x) + __uint_as_float(res.y);
        sm_out[j] = prev[j] + sm;
    }
}

// sub_col: subtract the per-width row-vector value from each column.
__device__ __forceinline__ void sub_col_att(
    float2 (&data)[2][ATT_WIDTH][8], const float (&val)[ATT_WIDTH]) {
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++) {
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                data[i][j][k].x -= val[j];
                data[i][j][k].y -= val[j];
            }
    }
}

// exp2 on one height-row of the att tile, across all ATT_WIDTH column-tiles.
__device__ __forceinline__ void exp2_base(float2 (&data)[ATT_WIDTH][8]) {
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            data[j][k].x = exp2f(data[j][k].x);
            data[j][k].y = exp2f(data[j][k].y);
        }
}

// mul_col on o_reg[4][8] by the per-width row-vector.
// o_reg (D x Q, col_l rt_32x32) has width == ATT_WIDTH; for Q_BLOCK_SIZE=32
// width is 1, so every D-subtile is scaled by val[0].
__device__ __forceinline__ void mul_col_o(float2 (&data)[4][ATT_WIDTH][8], const float (&val)[ATT_WIDTH]) {
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++)
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                data[i][j][k].x *= val[j];
                data[i][j][k].y *= val[j];
            }
}

// div_col on o_reg by the per-width row-vector.
__device__ __forceinline__ void div_col_o(float2 (&data)[4][ATT_WIDTH][8], const float (&val)[ATT_WIDTH]) {
    float inv[ATT_WIDTH];
    #pragma unroll
    for (int j = 0; j < ATT_WIDTH; j++) inv[j] = 1.0f / val[j];
    mul_col_o(data, inv);
}

// zero out o_reg (height x width x packed)
__device__ __forceinline__ void zero_o(float2 (&data)[4][ATT_WIDTH][8]) {
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < ATT_WIDTH; j++)
            #pragma unroll
            for (int k = 0; k < 8; k++)
                data[i][j][k] = {0.0f, 0.0f};
}

// zero out att_block (height x width x packed)
__device__ __forceinline__ void zero_att(float2 (&data)[2][ATT_WIDTH][8]) {
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < ATT_WIDTH; j++)
            #pragma unroll
            for (int k = 0; k < 8; k++)
                data[i][j][k] = {0.0f, 0.0f};
}

// ============================================================================
// Global-to-Shared: prefill swizzled offsets
// ============================================================================

// Shared tile parameters for st_bf<64, 128, st_32x32> (K)
// subtile: 32×32 bf16, subtile_bytes=2048, subtile_row_bytes=64
// subtiles_per_row=4, subtiles_per_col=2
// bytes_per_thread=16, memcpy_per_tile=2

// Shared tile parameters for st_bf<64, 128, st_8x32> (V)
// subtile: 8×32 bf16, subtile_bytes=512, subtile_row_bytes=64
// subtiles_per_row=4, subtiles_per_col=8
// bytes_per_thread=16, memcpy_per_tile=2

template<bool is_k>
__device__ __forceinline__ void prefill_offsets(
    uint32_t (&offsets)[2], int row_stride)
{
    constexpr int BYTES_PER_THREAD = 16;
    constexpr int BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE;

    constexpr int ST_ROWS = is_k ? 32 : 8;
    constexpr int ST_COLS = 32;
    constexpr int ST_ROW_BYTES = ST_COLS * 2;
    constexpr int ST_BYTES = ST_ROWS * ST_ROW_BYTES;
    constexpr int ST_PER_ROW = 128 / ST_COLS; // 4

    const int lid = laneid();
    const int wid = warpid();

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int lane_byte_off = (lid * BYTES_PER_THREAD) + (wid * BYTES_PER_WARP) +
                            (i * NUM_WARPS * BYTES_PER_WARP);
        int subtile_id = lane_byte_off / ST_BYTES;
        int subtile_row = subtile_id / ST_PER_ROW;
        int subtile_col = subtile_id % ST_PER_ROW;
        int sub_off = lane_byte_off % ST_BYTES;

        int row = sub_off / ST_ROW_BYTES;
        int col = (sub_off % ST_ROW_BYTES) / 2;

        uint32_t swizzled;
        if constexpr (is_k) {
            swizzled = swizzle_k(row, col);
        } else {
            swizzled = swizzle_v(row, col);
        }

        int sw_row = swizzled / ST_ROW_BYTES;
        int sw_col = (swizzled % ST_ROW_BYTES) / 2;

        int global_row = sw_row + subtile_row * ST_ROWS;
        int global_col = sw_col + subtile_col * ST_COLS;
        offsets[i] = (uint32_t)((global_row * row_stride + global_col) * 2);
    }
}

// ============================================================================
// Global-to-Shared: group load using buffer_load_lds
// ============================================================================
template<bool is_k>
__device__ __forceinline__ void group_load(
    bf16* smem_base,
    const int32_t s0, const int32_t s1,
    const bf16* global_base,
    int tile_idx, int row_stride,
    const uint32_t (&offsets)[2])
{
    constexpr int BYTES_PER_THREAD = 16;
    constexpr int BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE;
    constexpr int ELEMS_PER_WARP = BYTES_PER_WARP / 2;

    const int wid = warpid();

    const bf16* tile_ptr = global_base + tile_idx * KV_BLOCK_SIZE * row_stride;
    i32x4 srsrc = make_srsrc(tile_ptr, (uint32_t)(row_stride * KV_BLOCK_SIZE * 2));

    bf16* lds_base_ptr = smem_base + wid * ELEMS_PER_WARP;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        bf16* lds_elem_ptr = lds_base_ptr + i * NUM_WARPS * ELEMS_PER_WARP;
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, lds_ptr, BYTES_PER_THREAD,
            offsets[i], 0, 0, 0);
    }
}

// Version using pre-hoisted SGPR srsrc components.
// Reconstructs the strided base buffer resource (built once over the whole
// K/V tensor for this batch/head) from its four hoisted dwords, and selects
// the current KV tile via a scalar byte offset (SOFF) instead of rebuilding a
// per-tile srsrc. Mirrors the reference load() in kernel.cpp.
template<bool is_k>
__device__ __forceinline__ void group_load_srsrc(
    bf16* smem_base,
    int tile_idx, int row_stride,
    const uint32_t (&offsets)[2],
    int32_t s0, int32_t s1, int32_t s2, int32_t s3)
{
    constexpr int BYTES_PER_THREAD = 16;
    constexpr int BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE;
    constexpr int ELEMS_PER_WARP = BYTES_PER_WARP / 2;

    const int wid = warpid();

    // Rebuild the hoisted strided base srsrc {x,y,z,w}.
    i32x4 srsrc;
    srsrc.x = s0;
    srsrc.y = s1;
    srsrc.z = s2;
    srsrc.w = s3;

    // Scalar byte offset of this KV tile from the base pointer (tile 0).
    const uint32_t soff = (uint32_t)((size_t)tile_idx * KV_BLOCK_SIZE * row_stride * 2);

    bf16* lds_base_ptr = smem_base + wid * ELEMS_PER_WARP;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        bf16* lds_elem_ptr = lds_base_ptr + i * NUM_WARPS * ELEMS_PER_WARP;
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, lds_ptr, BYTES_PER_THREAD,
            offsets[i], soff, 0, 0);
    }
}

// ============================================================================
// Shared-to-Register: K load (row_l, rt_32x16 base, st_32x32 swizzle)
//   k_reg[2][8][4] from smem 64×128 bf16
// ============================================================================
__device__ __forceinline__ void load_k_from_shared(
    bf16_2 (&k_reg)[2][8][4], const bf16* smem)
{
    const int lid = laneid();
    const int row_offset = lid % 32;
    const int col_offset = 8 * (lid / 32);

    // st_32x32 subtile constants
    constexpr int ST_ROWS = 32;
    constexpr int ST_COLS = 32;
    constexpr int ST_BYTES = ST_ROWS * ST_COLS * 2;

    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(smem);

    // rt_32x16 row_l: base_tile_rows=32, base_tile_cols=16, stride=8
    // num_strides=1, register_subtiles_per_shared_subtile_row = 32/16 = 2
    // register_subtiles_per_shared_subtile_col = 32/32 = 1
    // subtiles_per_col=2, subtiles_per_row=4

    #pragma unroll
    for (int ii = 0; ii < 2; ii++) {        // subtiles_per_col
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {    // subtiles_per_row
            int shared_subtile_id = ii * 4 + jj;
            int offset = shared_subtile_id * ST_BYTES;

            #pragma unroll
            for (int j = 0; j < 2; j++) {    // register_subtiles_per_shared_subtile_row
                int row = row_offset;
                int col = j * 16 + col_offset;
                uint32_t swizzled = swizzle_k(row, col);
                uint32_t addr = src_ptr + swizzled;

                int register_row = ii;
                int register_col = jj * 2 + j;

                asm volatile(
                    "ds_read_b128 %0, %1 offset:%2\n"
                    : "=v"(*reinterpret_cast<float4*>(&k_reg[register_row][register_col][0]))
                    : "v"(addr), "i"(offset)
                    : "memory"
                );
            }
        }
    }
}

// ============================================================================
// Shared-to-Register: V load (col_l, rt_16x32_4 base, st_8x32 no-swizzle)
//   v_reg[4][4][4] from smem 64×128 bf16
// ============================================================================
__device__ __forceinline__ void load_v_from_shared(
    bf16_2 (&v_reg)[4][4][4], const bf16* smem)
{
    const int lid = laneid();

    // col_l lane mapping for rt_16x32_4
    const int row_offset = ((lid % 16) / 4) + ((lid / 32) * 4);
    const int col_offset = ((lid % 4) * 4) + (16 * ((lid % 32) / 16));

    // st_8x32 subtile constants
    constexpr int ST_ROWS = 8;
    constexpr int ST_COLS = 32;
    constexpr int ST_BYTES = ST_ROWS * ST_COLS * 2;
    constexpr int ST_PER_ROW = 4; // 128/32

    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(smem);

    // rt_16x32_4 col_l: base_tile_rows=16, base_tile_cols=32, stride=4
    // reductions=rows=16, threads_per_reduction=16/8=2, elements_per_stride_group=2*4=8
    // stride_groups_per_shared_subtile_col = 8/8 = 1
    // shared_subtiles_per_register_subtile_row = 32/32 = 1
    // shared_subtiles_per_register_subtile_col = 16/8 = 2
    // num_strides_inner = base_tile_num_strides / stride_groups = (8/4) / 1 = 2

    int col_in_sub = col_offset % ST_COLS;
    int shared_base_col = col_offset / ST_COLS; // always 0

    // l=0 (single stride group per shared subtile col)
    int row = row_offset;
    uint32_t sw = swizzle_v(row, col_in_sub);
    uint32_t addr = src_ptr + sw;

    #pragma unroll
    for (int k = 0; k < 2; k++) { // num_strides / stride_groups_per_shared_subtile_col
        int shared_base_row = k;

        #pragma unroll
        for (int i = 0; i < 4; i++) { // height = 4
            int shared_row = i * 2; // shared_subtiles_per_register_subtile_col = 2
            #pragma unroll
            for (int j = 0; j < 4; j++) { // width = 4
                int shared_col = j; // shared_subtiles_per_register_subtile_row = 1
                int shared_subtile_id = shared_row * ST_PER_ROW + shared_col;
                int off = shared_subtile_id * ST_BYTES + shared_base_row * ST_PER_ROW * ST_BYTES;

                int idx = k * 2; // k * stride / packing = k * 4 / 2

                asm volatile(
                    "ds_read_b64_tr_b16 %0, %1 offset:%2\n"
                    : "=v"(*reinterpret_cast<float2*>(&v_reg[i][j][idx]))
                    : "v"(addr), "i"(off)
                    : "memory"
                );
            }
        }
    }
}

// ============================================================================
// Global-to-Register: Q load (row_l, rt_32x16 base, bf16 global -> float reg)
//   q_reg_fl[8][4] as float2, from bf16 global tensor
// ============================================================================
__device__ __forceinline__ void load_q_global(
    float2 (&q_fl)[8][4],
    const bf16* Q_ptr, int row_stride, int batch, int tile, int head, int w)
{
    const int lid = laneid();
    const int row_offset = lid % 32;
    const int col_offset = 8 * (lid / 32);

    // rt_32x16 row_l with float: stride=8, elements_per_thread=8, packed_per_thread=4 (float2)
    // base_tile: rows=32, cols=16 (but with float, these are the shape params)
    // For float register tile: packed element is float2. stride/packing = 8/2 = 4.
    // num_strides = 1

    // w selects the 32-query column-tile within the Q block (one per ATT_WIDTH).
    const bf16* base = Q_ptr + (size_t)batch * ATTN_N * ATTN_H * ATTN_D
                     + (size_t)tile * Q_BLOCK_SIZE * ATTN_H * ATTN_D
                     + (size_t)w * 32 * ATTN_H * ATTN_D
                     + (size_t)head * ATTN_D;

    uint32_t buf_size = ATTN_B * ATTN_N * ATTN_H * ATTN_D * 2;
    buffer_resource br = make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)), buf_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

    // height=1, width=8 (32/32=1 height for 32-row tile, 128/16=8 width)
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = row_offset;
        int col = 16 * j + col_offset;

        // Load 8 bf16 (16 bytes) = buffer_load_b128
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(
            srsrc, (uint32_t)((row * row_stride + col) * 2), 0, 0);
        bf16_2* loaded = reinterpret_cast<bf16_2*>(&raw);

        // Convert bf16_2 -> float2
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            q_fl[j][l] = __bfloat1622float2(loaded[l]);
        }
    }
}

// ============================================================================
// Register-to-Global: O store (row_l after transpose, rt_32x32 base, float->bf16)
// ============================================================================
__device__ __forceinline__ void store_o_global(
    const bf16* O_ptr, const float2 (&o_out)[4][ATT_WIDTH][8],
    int row_stride, int batch, int tile, int head)
{
    const int lid = laneid();
    // row_l, rt_32x32 with float: rows=32, cols=32, stride=4
    // elements_per_stride_group = 8, num_strides = 4
    const int row_offset = lid % 32;
    const int col_offset = 4 * (lid / 32);

    const bf16* base = O_ptr + (size_t)batch * ATTN_N * ATTN_H * ATTN_D
                     + (size_t)tile * Q_BLOCK_SIZE * ATTN_H * ATTN_D
                     + (size_t)head * ATTN_D;

    uint32_t buf_size = ATTN_B * ATTN_N * ATTN_H * ATTN_D * 2;
    buffer_resource br = make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)), buf_size, 0x00020000);
    i32x4 srsrc = std::bit_cast<i32x4>(br);

    bf16_2 tmp[2];

    // height=1, width=4
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int row = row_offset;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int col = 32 * j + col_offset + k * 8;
            int idx = k * 2; // stride/packing = 4/2 = 2

            #pragma unroll
            for (int l = 0; l < 2; l++) {
                tmp[l] = __float22bfloat162_rn(o_out[j][0][idx + l]);
            }
            uint64_t val = *reinterpret_cast<uint64_t*>(tmp);
            llvm_amdgcn_raw_buffer_store_b64(
                val, srsrc,
                (uint32_t)((row * row_stride + col) * 2),
                0, 0);
        }
    }
}

// ============================================================================
// Register-to-Global: L_vec store (single float per warp)
// ============================================================================
__device__ __forceinline__ void store_lse_global(
    const float* L_ptr, float val,
    int L_dim3, int batch, int head, int tile, int w)
{
    const int lid = laneid();
    if (lid < 32) {
        // w selects the 32-query column-tile within the Q block.
        int seq_pos = tile * Q_BLOCK_SIZE + w * 32 + lid;
        const float* base = L_ptr + (size_t)batch * ATTN_H * 1 * L_dim3
                          + (size_t)head * 1 * L_dim3
                          + (size_t)seq_pos;
        uint32_t buf_size = ATTN_B * ATTN_H * 1 * L_dim3 * 4;
        buffer_resource br = make_buffer_resource(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)), buf_size, 0x00020000);
        i32x4 srsrc = std::bit_cast<i32x4>(br);
        llvm_amdgcn_raw_buffer_store_b32(
            __float_as_uint(val), srsrc, 0, 0, 0);
    }
}

// ============================================================================
// Kernel globals
// ============================================================================
struct attn_globals {
    bf16 *Q, *K, *V, *O;
    float *L;
    int Q_stride1, K_stride1, V_stride1, O_stride1;
    int L_dim3;
    hipStream_t stream;
};

// ============================================================================
// Main kernel
// ============================================================================
__launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals g) {

    extern __shared__ char __shm[];
    bf16* k_smem_0 = reinterpret_cast<bf16*>(__shm);
    bf16* k_smem_1 = k_smem_0 + KV_BLOCK_SIZE * ATTN_D;
    bf16* v_smem_0 = k_smem_1 + KV_BLOCK_SIZE * ATTN_D;
    bf16* v_smem_1 = v_smem_0 + KV_BLOCK_SIZE * ATTN_D;

    const int head_idx = (blockIdx.x % GROUP_SIZE) * GROUP_SIZE + (blockIdx.x / GROUP_SIZE);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    // Readfirstlane hoisting for buffer resources
    const bf16* k_base = g.K + (size_t)batch_idx * ATTN_N * ATTN_H_KV * ATTN_D
                       + (size_t)head_idx_kv * ATTN_D;
    const bf16* v_base = g.V + (size_t)batch_idx * ATTN_N * ATTN_H_KV * ATTN_D
                       + (size_t)head_idx_kv * ATTN_D;

    const int k_row_stride = g.K_stride1;
    const int v_row_stride = g.V_stride1;

    i32x4 k_srsrc_base = make_srsrc(k_base, (uint32_t)(k_row_stride * ATTN_N * 2), (uint32_t)(k_row_stride * 2));
    i32x4 v_srsrc_base = make_srsrc(v_base, (uint32_t)(v_row_stride * ATTN_N * 2), (uint32_t)(v_row_stride * 2));

    const int32_t ks0 = __builtin_amdgcn_readfirstlane(k_srsrc_base.x);
    const int32_t ks1 = __builtin_amdgcn_readfirstlane(k_srsrc_base.y);
    const int32_t ks2 = __builtin_amdgcn_readfirstlane(k_srsrc_base.z);
    const int32_t ks3 = __builtin_amdgcn_readfirstlane(k_srsrc_base.w);
    const int32_t vs0 = __builtin_amdgcn_readfirstlane(v_srsrc_base.x);
    const int32_t vs1 = __builtin_amdgcn_readfirstlane(v_srsrc_base.y); 
    const int32_t vs2 = __builtin_amdgcn_readfirstlane(v_srsrc_base.z);
    const int32_t vs3 = __builtin_amdgcn_readfirstlane(v_srsrc_base.w);

    constexpr float TEMPERATURE_SCALE = 0.08838834764f * 1.44269504089f;
    constexpr int num_tiles = NUM_KV_TILES;

    // Register tiles. ATT_WIDTH is the number of 32-wide query column-tiles in
    // the attention block (== Q_BLOCK_SIZE/32); the softmax row-vectors carry
    // one value per width slot, mirroring attn_tile::row_vec in kernel.cpp.
    bf16_2 q_reg[ATT_WIDTH][8][4];
    bf16_2 k_reg[2][8][4];
    bf16_2 k_reg_t[8][2][4];
    bf16_2 v_reg[4][4][4];
    float2 o_reg[4][ATT_WIDTH][8];
    float2 att_block[2][2][ATT_WIDTH][8]; // [double-buf idx][height][width][packed]
    bf16_2 att_block_bf16[2][ATT_WIDTH][8];
    bf16_2 att_block_bf16_in[ATT_WIDTH][4][4];
    float max_vec[ATT_WIDTH], norm_vec[ATT_WIDTH], max_vec_prev[ATT_WIDTH], scale_vec[ATT_WIDTH];

    zero_o(o_reg);
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) {
        norm_vec[w] = 0.0f;
        scale_vec[w] = 0.0f;
    }
    int pending_scale = 0;

    // Prefill swizzled offsets
    uint32_t swizzled_offsets_K[2];
    uint32_t swizzled_offsets_V[2];
    prefill_offsets<true>(swizzled_offsets_K, g.K_stride1);
    prefill_offsets<false>(swizzled_offsets_V, g.V_stride1);

    // Load K[0] into shared
    group_load_srsrc<true>(k_smem_0, 0, k_row_stride, swizzled_offsets_K,
                           ks0, ks1, ks2, ks3);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Load Q into registers, scale, convert to bf16 (one 32-wide tile per width).
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) {
        float2 q_reg_fl[8][4];
        load_q_global(q_reg_fl, g.Q, g.Q_stride1, batch_idx, tile_idx, head_idx, w);

        // Scale Q
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                q_reg_fl[j][l].x *= TEMPERATURE_SCALE;
                q_reg_fl[j][l].y *= TEMPERATURE_SCALE;
            }

        // Convert float -> bf16
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int l = 0; l < 4; l++)
                q_reg[w][j][l] = __float22bfloat162_rn(q_reg_fl[j][l]);
    }

    // Load K[1] into shared, V[0] into shared
    group_load_srsrc<true>(k_smem_1, 1, k_row_stride, swizzled_offsets_K,
                           ks0, ks1, ks2, ks3);
    group_load_srsrc<false>(v_smem_0, 0, v_row_stride, swizzled_offsets_V,
                            vs0, vs1, vs2, vs3);

    // Load K[0] from shared to registers
    load_k_from_shared(k_reg, k_smem_0);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // QK[0]
    zero_att(att_block[0]);
    transpose_k(k_reg_t, k_reg);
    mma_AtB_QK(att_block[0], k_reg_t, q_reg, att_block[0]);

    // Partial softmax for QK[0]
    col_max_reset(max_vec, att_block[0]);
    rv_copy(max_vec_prev, max_vec);
    rv_exp2(scale_vec);

    sub_col_att(att_block[0], max_vec);
    exp2_base(att_block[0][0]);
    __builtin_amdgcn_sched_barrier(0);
    mul_col_o(o_reg, scale_vec);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    __builtin_amdgcn_sched_barrier(0);

    // Load K[1] from shared, load K[2] into shared, load V[1] into shared
    load_k_from_shared(k_reg, k_smem_1);
    group_load_srsrc<true>(k_smem_0, 2, k_row_stride, swizzled_offsets_K,
                           ks0, ks1, ks2, ks3);
    group_load_srsrc<false>(v_smem_1, 1, v_row_stride, swizzled_offsets_V,
                            vs0, vs1, vs2, vs3);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // ========================================================================
    // Hot loop
    // ========================================================================
    #pragma unroll 2
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0: QK[odd]
        zero_att(att_block[1]);
        transpose_k(k_reg_t, k_reg);
        if (pending_scale) {
            rv_mul(norm_vec, scale_vec);
        }
        mma_AtB_QK(att_block[1], k_reg_t, q_reg, att_block[1]);
        // Finish softmax for QK[even]
        exp2_base(att_block[0][1]);
        col_sum_accum(norm_vec, att_block[0], norm_vec);
        copy_f32_to_bf16(att_block_bf16, att_block[0]);
        reinterpret_att(att_block_bf16_in, att_block_bf16);
        sched_barrier_exp_pairs<6, 3, 1>();
        sched_barrier_pairs<10, 5, 1>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: Load K[j] into shared, load V from shared
        group_load_srsrc<true>(k_smem_1, j, k_row_stride, swizzled_offsets_K,
                               ks0, ks1, ks2, ks3);
        load_v_from_shared(v_reg, v_smem_0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: A[even]*V, partial softmax for QK[odd]
        __builtin_amdgcn_s_setprio(1);
        mma_AtB_OV(o_reg, v_reg, att_block_bf16_in, o_reg);
        col_max_accum(max_vec, att_block[1], max_vec_prev);
        sched_barrier_pairs<4, 5, 2>();
        {
            int all_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD);
            if (__builtin_expect(all_ok, 1)) {
                rv_copy(max_vec, max_vec_prev);
                pending_scale = 0;
            } else {
                rv_diff_exp2(scale_vec, max_vec_prev, max_vec);
                mul_col_o(o_reg, scale_vec);
                rv_copy(max_vec_prev, max_vec);
                pending_scale = 1;
            }
        }
        sub_col_att(att_block[1], max_vec);
        exp2_base(att_block[1][0]);
        sched_barrier_pairs<6, 5, 2>();
        sched_barrier_exp_pairs<6, 3, 2>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3: Load V[j-1] into shared, load K from shared
        group_load_srsrc<false>(v_smem_0, j - 1, v_row_stride, swizzled_offsets_V,
                                vs0, vs1, vs2, vs3);
        load_k_from_shared(k_reg, k_smem_0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: QK[even]
        zero_att(att_block[0]);
        transpose_k(k_reg_t, k_reg);
        if (pending_scale) {
            rv_mul(norm_vec, scale_vec);
        }
        mma_AtB_QK(att_block[0], k_reg_t, q_reg, att_block[0]);
        // Finish softmax for QK[odd]
        exp2_base(att_block[1][1]);
        col_sum_accum(norm_vec, att_block[1], norm_vec);
        copy_f32_to_bf16(att_block_bf16, att_block[1]);
        reinterpret_att(att_block_bf16_in, att_block_bf16);
        sched_barrier_exp_pairs<6, 3, 3>();
        sched_barrier_pairs<10, 5, 3>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5: Load K[j+1] into shared, load V from shared
        group_load_srsrc<true>(k_smem_0, j + 1, k_row_stride, swizzled_offsets_K,
                               ks0, ks1, ks2, ks3);
        load_v_from_shared(v_reg, v_smem_1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: A[odd]*V, partial softmax for QK[even]
        __builtin_amdgcn_s_setprio(1);
        mma_AtB_OV(o_reg, v_reg, att_block_bf16_in, o_reg);
        col_max_accum(max_vec, att_block[0], max_vec_prev);
        sched_barrier_pairs<4, 5, 4>();
        {
            int all_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD);
            if (__builtin_expect(all_ok, 1)) {
                rv_copy(max_vec, max_vec_prev);
                pending_scale = 0;
            } else {
                rv_diff_exp2(scale_vec, max_vec_prev, max_vec);
                mul_col_o(o_reg, scale_vec);
                rv_copy(max_vec_prev, max_vec);
                pending_scale = 1;
            }
        }
        sub_col_att(att_block[0], max_vec);
        exp2_base(att_block[0][0]);
        sched_barrier_pairs<6, 5, 4>();
        sched_barrier_exp_pairs<6, 3, 4>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7: Load V[j] into shared, load K from shared
        group_load_srsrc<false>(v_smem_1, j, v_row_stride, swizzled_offsets_V,
                                vs0, vs1, vs2, vs3);
        load_k_from_shared(k_reg, k_smem_1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // ========================================================================
    // Epilogue
    // ========================================================================

    // Cluster 0: QK[last odd]
    zero_att(att_block[1]);
    transpose_k(k_reg_t, k_reg);
    mma_AtB_QK(att_block[1], k_reg_t, q_reg, att_block[1]);
    // Finish softmax for QK[last even]
    exp2_base(att_block[0][1]);
    if (pending_scale) {
        rv_mul(norm_vec, scale_vec);
    }
    col_sum_accum(norm_vec, att_block[0], norm_vec);
    copy_f32_to_bf16(att_block_bf16, att_block[0]);
    reinterpret_att(att_block_bf16_in, att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 5>();
    sched_barrier_pairs<10, 5, 5>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1: Load K[num_tiles-1] into shared, load V from shared
    group_load_srsrc<true>(k_smem_1, num_tiles - 1, k_row_stride, swizzled_offsets_K,
                           ks0, ks1, ks2, ks3);
    load_v_from_shared(v_reg, v_smem_0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2: A*V, partial softmax
    __builtin_amdgcn_s_setprio(1);
    mma_AtB_OV(o_reg, v_reg, att_block_bf16_in, o_reg);
    col_max_accum(max_vec, att_block[1], max_vec_prev);
    rv_diff_exp2(scale_vec, max_vec_prev, max_vec);
    rv_copy(max_vec_prev, max_vec);
    sub_col_att(att_block[1], max_vec);
    exp2_base(att_block[1][0]);
    sched_barrier_pairs<10, 5, 6>();
    sched_barrier_exp_pairs<6, 3, 6>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col_o(o_reg, scale_vec);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3: Load V[num_tiles-2] into shared, load K from shared
    group_load_srsrc<false>(v_smem_0, num_tiles - 2, v_row_stride, swizzled_offsets_V,
                            vs0, vs1, vs2, vs3);
    load_k_from_shared(k_reg, k_smem_0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4: QK
    zero_att(att_block[0]);
    transpose_k(k_reg_t, k_reg);
    mma_AtB_QK(att_block[0], k_reg_t, q_reg, att_block[0]);
    // Finish softmax
    exp2_base(att_block[1][1]);
    rv_mul(norm_vec, scale_vec);
    col_sum_accum(norm_vec, att_block[1], norm_vec);
    copy_f32_to_bf16(att_block_bf16, att_block[1]);
    reinterpret_att(att_block_bf16_in, att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 7>();
    sched_barrier_pairs<10, 5, 7>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5: Load V from shared
    load_v_from_shared(v_reg, v_smem_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6: A*V, partial softmax
    __builtin_amdgcn_s_setprio(1);
    mma_AtB_OV(o_reg, v_reg, att_block_bf16_in, o_reg);
    col_max_accum(max_vec, att_block[0], max_vec_prev);
    rv_diff_exp2(scale_vec, max_vec_prev, max_vec);
    rv_copy(max_vec_prev, max_vec);
    sub_col_att(att_block[0], max_vec);
    exp2_base(att_block[0][0]);
    sched_barrier_pairs<10, 5, 8>();
    sched_barrier_exp_pairs<6, 3, 8>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col_o(o_reg, scale_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7: Load V[num_tiles-1] into shared, load K from shared
    group_load_srsrc<false>(v_smem_1, num_tiles - 1, v_row_stride, swizzled_offsets_V,
                            vs0, vs1, vs2, vs3);
    load_k_from_shared(k_reg, k_smem_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8: QK
    zero_att(att_block[1]);
    transpose_k(k_reg_t, k_reg);
    mma_AtB_QK(att_block[1], k_reg_t, q_reg, att_block[1]);
    // Finish softmax
    exp2_base(att_block[0][1]);
    rv_mul(norm_vec, scale_vec);
    col_sum_accum(norm_vec, att_block[0], norm_vec);
    copy_f32_to_bf16(att_block_bf16, att_block[0]);
    reinterpret_att(att_block_bf16_in, att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 9>();
    sched_barrier_pairs<10, 5, 9>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9: Load V from shared
    load_v_from_shared(v_reg, v_smem_0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10: A*V, full softmax for last QK
    mma_AtB_OV(o_reg, v_reg, att_block_bf16_in, o_reg);
    col_max_accum(max_vec, att_block[1], max_vec_prev);
    rv_diff_exp2(scale_vec, max_vec_prev, max_vec);
    rv_copy(max_vec_prev, max_vec);
    sub_col_att(att_block[1], max_vec);
    exp2_base(att_block[1][0]);
    sched_barrier_pairs<10, 5, 10>();
    sched_barrier_exp_pairs<6, 3, 10>();
    __builtin_amdgcn_sched_barrier(0);

    exp2_base(att_block[1][1]);
    rv_mul(norm_vec, scale_vec);
    col_sum_accum(norm_vec, att_block[1], norm_vec);
    copy_f32_to_bf16(att_block_bf16, att_block[1]);
    reinterpret_att(att_block_bf16_in, att_block_bf16);
    __builtin_amdgcn_sched_barrier(0);
    mul_col_o(o_reg, scale_vec);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11: Load V from shared
    load_v_from_shared(v_reg, v_smem_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12: Final A*V and normalize
    mma_AtB_OV(o_reg, v_reg, att_block_bf16_in, o_reg);
    div_col_o(o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    // Store O
    store_o_global(g.O, o_reg, g.O_stride1, batch_idx, tile_idx, head_idx);

    // Compute and store LSE (one per query column-tile)
    #pragma unroll
    for (int w = 0; w < ATT_WIDTH; w++) {
        float lse_max = max_vec[w] * 0.69314718056f;
        float lse = logf(norm_vec[w]) + lse_max;
        store_lse_global(g.L, lse, g.L_dim3, batch_idx, head_idx, tile_idx, w);
    }
}

// ============================================================================
// Dispatch
// ============================================================================
void dispatch_micro(attn_globals g) {
    dim3 grid(ATTN_H,
              (ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS,
              ATTN_B);
    dim3 block(NUM_THREADS);
    size_t smem_size = 4 * KV_BLOCK_SIZE * ATTN_D * sizeof(bf16);
    hipFuncSetAttribute((void*)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    attend_ker<<<grid, block, smem_size, g.stream>>>(g);
}

// ============================================================================
// Pybind11 module
// ============================================================================
PYBIND11_MODULE(hip_tk_kernel, m) {
    m.doc() = "hip_tk_kernel python module (plain HIP)";
    m.def("dispatch_micro", [](py::object q_obj, py::object k_obj, py::object v_obj,
                                py::object o_obj, py::object l_obj) {
        auto get_tensor = [](py::object obj, const char* name) -> std::pair<void*, std::array<int,4>> {
            if (!py::hasattr(obj, "__class__") ||
                obj.attr("__class__").attr("__name__").cast<std::string>() != "Tensor") {
                throw std::runtime_error(std::string(name) + " must be a torch.Tensor");
            }
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error(std::string(name) + " must be contiguous");
            }
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error(std::string(name) + " must be on CUDA device");
            }

            std::array<int,4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<py::tuple>();
            size_t dims = py_shape.size();
            for (size_t i = 0; i < dims && i < 4; ++i) {
                shape[4 - dims + i] = py::cast<int>(py_shape[i]);
            }

            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();
            return {reinterpret_cast<void*>(data_ptr), shape};
        };

        auto [q_ptr, q_shape] = get_tensor(q_obj, "Q");
        auto [k_ptr, k_shape] = get_tensor(k_obj, "K");
        auto [v_ptr, v_shape] = get_tensor(v_obj, "V");
        auto [o_ptr, o_shape] = get_tensor(o_obj, "O");
        auto [l_ptr, l_shape] = get_tensor(l_obj, "L");

        attn_globals g;
        g.Q = reinterpret_cast<bf16*>(q_ptr);
        g.K = reinterpret_cast<bf16*>(k_ptr);
        g.V = reinterpret_cast<bf16*>(v_ptr);
        g.O = reinterpret_cast<bf16*>(o_ptr);
        g.L = reinterpret_cast<float*>(l_ptr);

        // stride<1> = shape[2] * shape[3]
        g.Q_stride1 = q_shape[2] * q_shape[3];
        g.K_stride1 = k_shape[2] * k_shape[3];
        g.V_stride1 = v_shape[2] * v_shape[3];
        g.O_stride1 = o_shape[2] * o_shape[3];
        g.L_dim3 = l_shape[3];
        g.stream = 0;

        dispatch_micro(g);
    });
}
