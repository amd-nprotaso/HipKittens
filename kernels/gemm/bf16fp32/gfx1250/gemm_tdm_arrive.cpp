/**
 * @file gemm_tdm_arrive.cpp
 * @brief Rung 8 -- per-transfer LDS-barrier TDM GEMM for gfx1250.
 * @experimental
 *
 * Demonstrates the `load_tdm_arrive` / `barrier_lds` / `wait_barrier` triple
 * end to end. The library encoding of the D#'s `atomic_barrier_enable` bit
 * and the `atomic_barrier_address` field width are derived from the gfx1250
 * spec summary; the exact bit positions still need cross-checking against
 * the SP3 reference, and the auto-arrive path may not be modelled by every
 * runtime. The kernel is excluded from the default ladder smoke-test sweep
 * (`run_all.sh`) until that verification lands.
 *
 * Diff vs `gemm_expert`: replace cooperative async loads with `load_tdm_arrive`
 * issued by wave 0 (for A) and wave 1 (for B). Each TDM transfer is paired
 * with its own `barrier_lds` cell so the consumer waits on a phase flip
 * specific to that operand instead of draining the global `tensorcnt`.
 *
 * Exercises the new fine-grained API:
 *   - `sync::barrier_lds`             -- 64-bit LDS barrier cell.
 *   - `sync::init_barrier(bar, n)`    -- prime the cell for `n` arrivals.
 *   - `kittens::load_tdm_arrive(..., bar)` -- TDM that auto-arrives on `bar`.
 *   - `sync::wait_barrier(bar, phase)`-- block on the cell's phase flip.
 *
 * The kernel proves out two things:
 *   1. `load_tdm_arrive` constructs a valid D# with `atomic_barrier_enable`
 *      set and the LDS barrier address routed in, and the runtime delivers
 *      the auto-arrive correctly.
 *   2. Independent phases on A_bar and B_bar let the kernel keep more than
 *      one TDM transfer in flight at a time without inter-operand stalls.
 *
 * Tile: 64x64 output, K_STEP = 32, 4 warps in a 2x2 layout (matches the
 * rest of the ladder).
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

using Pad = lds_pad_default;
constexpr int A_ELEMS_PAD = Pad::padded_elems(BLOCK_M * K_STEP);
constexpr int B_ELEMS_PAD = Pad::padded_elems(BLOCK_N * K_STEP);

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_tdm_arrive_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    // Segment 0 layout: A slabs followed by the four 8-byte barrier cells.
    // Allocate barriers FIRST in segment 0 (their addresses fit in 16 bits,
    // which is what the D# `atomic_barrier_address` field carries), then
    // the A buffers; finally B in segment 1.
    sync::barrier_lds(&A_bar)[2] = al.allocate_in<segment<0>, sync::barrier_lds, 2>();
    sync::barrier_lds(&B_bar)[2] = al.allocate_in<segment<0>, sync::barrier_lds, 2>();
    bf16(&A_lds)[2][A_ELEMS_PAD] = al.allocate_in<segment<0>, bf16, 2, A_ELEMS_PAD>();
    bf16(&B_lds)[2][B_ELEMS_PAD] = al.allocate_in<segment<1>, bf16, 2, B_ELEMS_PAD>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    // One thread primes the four cells. Each cell expects 1 arrival per
    // phase (the single TDM transfer that will target it).
    if (threadIdx.x == 0) {
        sync::init_barrier(&A_bar[0].state, 1);
        sync::init_barrier(&A_bar[1].state, 1);
        sync::init_barrier(&B_bar[0].state, 1);
        sync::init_barrier(&B_bar[1].state, 1);
    }
    sync::sync();

    sched::expert _sched;

    // Per-buffer parity bits. The cell's phase bit starts at 0 and flips
    // each time the pending count drains; `wait_barrier(.., phase ^ 1)`
    // unblocks once the next arrival lands.
    int A_phase[2] = {0, 0};
    int B_phase[2] = {0, 0};

    // Prologue: wave 0 issues A[0], wave 1 issues B[0].
    if (wid == 0) {
        g2s::load_tdm_arrive<Pad, BLOCK_M, K_STEP>(
            A_lds[0], g.a, {0, 0, tile_m, 0}, M, K, K, &A_bar[0].state);
    }
    if (wid == 1) {
        g2s::load_tdm_arrive<Pad, BLOCK_N, K_STEP>(
            B_lds[0], g.b, {0, 0, tile_n, 0}, N, K, K, &B_bar[0].state);
    }

    for (int k = 0; k < k_iters; ++k) {
        const int cur = k & 1, nxt = 1 - cur;

        // Issue the next K-step into the inactive buffer (independent
        // barriers so A and B don't serialize on each other).
        if (k + 1 < k_iters) {
            if (wid == 0) {
                g2s::load_tdm_arrive<Pad, BLOCK_M, K_STEP>(
                    A_lds[nxt], g.a, {0, 0, tile_m, k + 1}, M, K, K,
                    &A_bar[nxt].state);
            }
            if (wid == 1) {
                g2s::load_tdm_arrive<Pad, BLOCK_N, K_STEP>(
                    B_lds[nxt], g.b, {0, 0, tile_n, k + 1}, N, K, K,
                    &B_bar[nxt].state);
            }
        }

        // Wait for THIS K-step's transfers (independent of the next).
        // Toggle the parity for the cell we're about to consume.
        A_phase[cur] ^= 1;
        B_phase[cur] ^= 1;
        sync::wait_barrier(&A_bar[cur].state, A_phase[cur]);
        sync::wait_barrier(&B_bar[cur].state, B_phase[cur]);
        sync::sync();   // make A/B-arrived state visible to every consumer warp

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        kittens::load_b128<Pad, WARP_M, K_STEP>(
            A_reg, A_lds[cur] + Pad::padded(warp_r * WARP_M * K_STEP));
        kittens::load_b128<Pad, WARP_N, K_STEP>(
            B_reg, B_lds[cur] + Pad::padded(warp_c * WARP_N * K_STEP));

        sync::wait_ds();
        mma_ABt_burst(C_acc, A_reg, B_reg, C_acc);

        sync::sync();
    }

    bf16* c_base = reinterpret_cast<bf16*>(&g.c[{0, 0, 0, 0}]);
    store_acc<WARP_M / 16, WARP_N / 16>(
        c_base,
        tile_m * BLOCK_M + warp_r * WARP_M,
        tile_n * BLOCK_N + warp_c * WARP_N,
        N, C_acc);
}

void dispatch(gemm_globals g)
{
    // Same layout as `gemm_segment`/`gemm_expert` (A in seg 0, B in seg 1)
    // plus 4 barrier cells in seg 0.
    constexpr size_t bar_bytes = 4 * sizeof(sync::barrier_lds);
    const size_t mem_size = LDS_SEGMENT_BYTES + 2 * B_ELEMS_PAD * sizeof(bf16);
    (void)bar_bytes;
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_tdm_arrive_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    gemm_tdm_arrive_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
