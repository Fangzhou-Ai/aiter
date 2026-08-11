// SPDX-License-Identifier: MIT
//
// Standalone H40-derived Kimi-K3 target-verify prototype.
//
// This deliberately lives outside the production dispatch while its numerical
// and performance contract is being established.  Python drives the exported
// C ABI with torch tensor pointers.

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cstdint>

#ifndef __launch_bounds__
#define __launch_bounds__(threads, blocks) \
    __attribute__((amdgpu_flat_work_group_size(1, threads), \
                   amdgpu_waves_per_eu(blocks)))
#endif

namespace kimi_verify_h40 {

#ifndef KIMI_H40_PV_BATCH
#define KIMI_H40_PV_BATCH 4
#endif
#ifndef KIMI_H40_EP_PITCH
#define KIMI_H40_EP_PITCH 520
#endif
#ifndef KIMI_H40_Q_SUB
#define KIMI_H40_Q_SUB 2
#endif
// Ablation probes -- ALL give WRONG results.  They exist to price a component
// by removing it and seeing how much time comes back.
// Stage KV through registers instead of buffer_load...lds.  A microbenchmark
// at the same 3.6 MB/WG footprint measures the direct-to-LDS DMA path at
// 14.4 GB/s/CU against 23.3 GB/s/CU for plain register loads at 8 in flight --
// the DMA path is latency-limited well below what the CU can do.  Since the
// kernel time equals its DMA time exactly (MFMA is fully hidden inside it),
// closing that gap should drop it onto the MFMA floor: 249.7 -> ~178 us.
// STAGE_BATCH = halves issued before their ds_writes, i.e. loads in flight
// = 5 * STAGE_BATCH, at 4 VGPRs each.
#ifndef KIMI_H40_SCHED
#define KIMI_H40_SCHED 0
#endif
#ifndef KIMI_H40_QK_PIPE
#define KIMI_H40_QK_PIPE 2
#endif
#ifndef KIMI_H40_REDUCE_THREADS
#define KIMI_H40_REDUCE_THREADS 128
#endif

struct kargs
{
    const void* __restrict__ q_ptr;       // [T, qlen*nhead, 576] fp8
    const void* __restrict__ kv_ptr;      // [rows, 576] fp8
    const int* __restrict__ kv_indptr;    // [T+1]
    const int* __restrict__ kv_indices;   // [sum(seq_len)]
    float* __restrict__ part_m;           // [T, splits, rows_pad]
    float* __restrict__ part_l;           // [T, splits, rows_pad]
    float* __restrict__ part_acc;         // [T, splits, rows_pad, 512]
    void* __restrict__ out_ptr;           // [T, qlen*nhead, 512] bf16
    int T;
    int qlen;
    int nhead;
    int valid_rows;
    int rows_pad;
    int h_offset;
    int kv_rows;
    int num_splits;
    float qk_scale_log2;
    float output_scale;
};

struct traits
{
    static constexpr int Q_TILE = 16;
    static constexpr int Q_SUB = KIMI_H40_Q_SUB;
    static constexpr int NUM_WARPS = 4;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int HEADS_PER_BLOCK = NUM_WARPS * Q_SUB * Q_TILE;

    static constexpr int KV_TILE = 128;
    static constexpr int D_QK = 576;
    static constexpr int D_V = 512;
    static constexpr int D_SLICES = 5; // 4x128 + a zero-padded 64 tail
    static constexpr int N_TILES = KV_TILE / 16;
    static constexpr int O_TILES = D_V / 16;
    static constexpr int K_BLOCKS = KV_TILE / 32;
    static constexpr int S_LEN = N_TILES * 4;

    // Same token permutation as pa_sparse_prefill_fp8_h40.  The first 32
    // 16-byte cells are V and the final four cells are K-only RoPE.
    static constexpr int ATOM = 128;
    static constexpr int PAD = 64;
    static constexpr int ROW = 36 * ATOM + PAD;
    static constexpr int TILE_BYTES = 16 * ROW;
    static constexpr int SLOTS_PER_WAVE = 16 / NUM_WARPS;
    static constexpr int SMEM_BYTES = 2 * TILE_BYTES;

    static constexpr int PV_BATCH = KIMI_H40_PV_BATCH;
    static constexpr int QK_PIPE = KIMI_H40_QK_PIPE;
    static_assert(QK_PIPE == 1 || QK_PIPE == 2);
    static_assert(O_TILES % PV_BATCH == 0);
    static constexpr int EP_PITCH =
        KIMI_H40_EP_PITCH; // fp32 elements per staged head row
    static_assert(SMEM_BYTES <= 163840);
    static_assert(
        NUM_WARPS * Q_TILE * EP_PITCH * (int)sizeof(float)
        <= SMEM_BYTES);
};

__host__ __device__ inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void partial_kernel(kargs args);
__global__ void reduce_kernel(kargs args);

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

namespace device {

using namespace opus;
using fp8_t = _BitInt(8);
using u8_t = unsigned char;
using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x4 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

constexpr float LOWEST = -3.4028234663852886e38f;

__device__ inline f32x4 mma_unit(const i32x8& a,
                                const i32x8& b,
                                const f32x4& c)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 0, 0, 0, 127, 0, 127);
}

__device__ inline f32x4 mma_p(const i32x8& v,
                             const i32x8& p,
                             const f32x4& c,
                             int p_scale)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        v, p, c, 0, 0, 0, 127, 0, p_scale);
}

__device__ inline i32x8 pack32(const i32x4& lo, const i32x4& hi)
{
    i32x8 out;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        out[i] = lo[i];
        out[4 + i] = hi[i];
    }
    return out;
}

__device__ inline float reduce_max_groups(float v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    v = max(std::bit_cast<float>(r32.x), std::bit_cast<float>(r32.y));
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    return max(std::bit_cast<float>(r16.x), std::bit_cast<float>(r16.y));
}

__device__ inline float reduce_sum_groups(float v)
{
    vector_t<u32_t, 2> r32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    v = std::bit_cast<float>(r32.x) + std::bit_cast<float>(r32.y);
    vector_t<u32_t, 2> r16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<u32_t>(v), std::bit_cast<u32_t>(v), false, true);
    return std::bit_cast<float>(r16.x) + std::bit_cast<float>(r16.y);
}

__device__ inline void lds_barrier()
{
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
}

} // namespace device

__global__ __launch_bounds__(traits::BLOCK_SIZE, 1)
void partial_kernel(kargs args)
{
    using namespace opus;
    using namespace device;
    using T = traits;

    const int req = (int)__builtin_amdgcn_workgroup_id_x();
    const int h_block = (int)__builtin_amdgcn_workgroup_id_y();
    const int split = (int)__builtin_amdgcn_workgroup_id_z();
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    int lane = tid & 63;
    asm volatile("" : "+v"(lane));
    const int warp = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int c = lane & 15;
    const int g = lane >> 4;

    const int full_begin = args.kv_indptr[req];
    const int full_len = args.kv_indptr[req + 1] - full_begin;
    const int full_tiles = ceil_div(full_len, T::KV_TILE);
    const int tiles_per_split = ceil_div(full_tiles, args.num_splits);
    const int first_tile = split * tiles_per_split;
    const int last_tile = min(first_tile + tiles_per_split, full_tiles);
    const int segment_begin = first_tile * T::KV_TILE;
    const int segment_end = min(last_tile * T::KV_TILE, full_len);
    const int segment_len = max(0, segment_end - segment_begin);
    const int num_tiles = ceil_div(segment_len, T::KV_TILE);

    const int h_base =
        args.h_offset + h_block * T::HEADS_PER_BLOCK;
    const int h_wave = warp * (T::Q_SUB * T::Q_TILE);

    __shared__ char smem[T::SMEM_BYTES];

    auto g_q = make_gmem(
        reinterpret_cast<const fp8_t*>(args.q_ptr)
            + ((size_t)req * args.valid_rows + h_base) * T::D_QK,
        (unsigned)max(0, args.valid_rows - h_base) * T::D_QK);

    i32x8 v_q[T::Q_SUB][T::D_SLICES];
    const i32x4 zero4{0, 0, 0, 0};
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
    {
        const int q_row = (h_wave + qs * T::Q_TILE + c) * T::D_QK;
        const int q_off = q_row + 16 * g;
#pragma unroll
        for(int ds = 0; ds < T::D_SLICES; ++ds)
        {
            const i32x4 lo = __builtin_bit_cast(
                i32x4, load<16>(g_q, q_off + ds * 128));
            const i32x4 hi = ds == T::D_SLICES - 1
                ? zero4
                : __builtin_bit_cast(
                      i32x4, load<16>(g_q, q_off + ds * 128 + 64));
            v_q[qs][ds] = pack32(lo, hi);
        }
    }

    f32x4 v_o[T::Q_SUB][T::O_TILES];
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
        for(int ot = 0; ot < T::O_TILES; ++ot)
            v_o[qs][ot] = f32x4{0.f, 0.f, 0.f, 0.f};

    float m_row[T::Q_SUB];
    float m_ref[T::Q_SUB];
    float l_row[T::Q_SUB];
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
    {
        m_row[qs] = LOWEST;
        m_ref[qs] = LOWEST;
        l_row[qs] = 0.f;
    }

    if(num_tiles > 0)
    {
        auto g_kv = make_gmem(
            reinterpret_cast<const fp8_t*>(args.kv_ptr),
            (unsigned)args.kv_rows * T::D_QK);
        auto g_idx = make_gmem(
            args.kv_indices + full_begin + segment_begin,
            (unsigned)segment_len * sizeof(int));

        const int qk_base =
            (c >> 2) * T::ROW + (c & 3) * 16 + g * T::ATOM;
        const int pv_base = g * T::ROW + c * 8;
        const int p_lane = 16 * ((lane & 7) >> 2) + (lane & 3);
        const int d_lane = (lane >> 3) * 16;

        auto load_rows = [&](int tile_idx,
                             int (&rows)[T::SLOTS_PER_WAVE]) {
#pragma unroll
            for(int half = 0; half < T::SLOTS_PER_WAVE; ++half)
            {
                const int s_hi = T::SLOTS_PER_WAVE * warp + half;
                const int p =
                    32 * (s_hi >> 2) + 4 * (s_hi & 3) + p_lane;
                rows[half] =
                    load(g_idx, tile_idx * T::KV_TILE + p)[0];
            }
        };

        auto issue_copy = [&](const int (&rows)[T::SLOTS_PER_WAVE],
                              int buf) {
            char* const dst = smem + buf * T::TILE_BYTES;
#pragma unroll
            for(int half = 0; half < T::SLOTS_PER_WAVE; ++half)
            {
                const int s_hi = T::SLOTS_PER_WAVE * warp + half;
                const unsigned goff =
                    (unsigned)rows[half] * T::D_QK
                    + (unsigned)d_lane;
#pragma unroll
                for(int q = 0; q < 4; ++q)
                    g_kv.template async_load<16>(
                        dst + s_hi * T::ROW + q * 1024,
                        goff + (unsigned)q * 128u);
                if(d_lane < 64)
                    g_kv.template async_load<16>(
                        dst + s_hi * T::ROW + 4 * 1024,
                        goff + 4u * 128u);
            }
        };

        int rows0[T::SLOTS_PER_WAVE];
        int rows1[T::SLOTS_PER_WAVE];
        load_rows(0, rows0);
        issue_copy(rows0, 0);
        load_rows(1, rows0);

        int buf = 0;
        for(int tile = 0; tile < num_tiles; ++tile)
        {
            s_waitcnt_vmcnt(0_I);
            lds_barrier();

            const bool has_next = tile + 1 < num_tiles;
            if(has_next)
            {
                issue_copy(rows0, buf ^ 1);
                load_rows(tile + 2, rows1);
            }

            auto s_kv = make_smem(
                reinterpret_cast<fp8_t*>(
                    smem + buf * T::TILE_BYTES));

            float scores[T::Q_SUB][T::S_LEN];

            auto load_k = [&](auto i_nt,
                              i32x8 (&kk)[T::D_SLICES]) {
                constexpr int nt = decltype(i_nt)::value;
                constexpr int nt_off =
                    (nt >> 1) * 4 * T::ROW + (nt & 1) * 64;
                static_for<T::D_SLICES>([&](auto i_ds) {
                    constexpr int ds = i_ds.value;
                    constexpr int off0 = nt_off + ds * 1024;
                    const i32x4 lo = __builtin_bit_cast(
                        i32x4,
                        s_kv.template _load<16>(qk_base + off0));
                    const i32x4 hi =
                        ds == T::D_SLICES - 1
                        ? zero4
                        : __builtin_bit_cast(
                              i32x4,
                              s_kv.template _load<16>(
                                  qk_base + off0 + 4 * T::ATOM));
                    kk[ds] = pack32(lo, hi);
                });
            };

            i32x8 kbuf[T::QK_PIPE][T::D_SLICES];
            if constexpr(T::QK_PIPE == 2)
                load_k(number<0>{}, kbuf[0]);
            static_for<T::N_TILES>([&](auto i_nt) {
                constexpr int nt = i_nt.value;
                if constexpr(T::QK_PIPE == 2)
                {
                    if constexpr(nt + 1 < T::N_TILES)
                        load_k(number<nt + 1>{}, kbuf[(nt + 1) & 1]);
                }
                else
                    load_k(number<nt>{}, kbuf[0]);

                f32x4 acc[T::Q_SUB];
#pragma unroll
                for(int qs = 0; qs < T::Q_SUB; ++qs)
                    acc[qs] = f32x4{0.f, 0.f, 0.f, 0.f};
                static_for<T::D_SLICES>([&](auto i_ds) {
                    constexpr int ds = i_ds.value;
#pragma unroll
                    for(int qs = 0; qs < T::Q_SUB; ++qs)
                        acc[qs] = mma_unit(
                            kbuf[T::QK_PIPE == 2 ? (nt & 1) : 0][ds],
                            v_q[qs][ds],
                            acc[qs]);
                });
#pragma unroll
                for(int qs = 0; qs < T::Q_SUB; ++qs)
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        scores[qs][nt * 4 + i] = acc[qs][i];
#if KIMI_H40_SCHED
#pragma unroll
                for(int rep = 0; rep < 3; ++rep)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
                    __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);
                }
#endif
            });

            const int tile_global =
                segment_begin + tile * T::KV_TILE;
            const bool maybe_mask =
                tile_global + T::KV_TILE - 1 + args.qlen - 1
                >= full_len;

            i32x8 v_p[T::Q_SUB];
            float bias[T::Q_SUB];
            int dexp[T::Q_SUB];
#pragma unroll
            for(int qs = 0; qs < T::Q_SUB; ++qs)
            {
                const int row =
                    h_base + h_wave + qs * T::Q_TILE + c;
                const int qpos = row / args.nhead;
                const int shift = args.qlen - 1 - qpos;

                float local_max = LOWEST;
#pragma unroll
                for(int a = 0; a < T::N_TILES; ++a)
#pragma unroll
                    for(int b = 0; b < 4; ++b)
                    {
                        float s = scores[qs][a * 4 + b];
                        const int kv_pos =
                            tile_global + 16 * a + 4 * g + b;
                        if(maybe_mask && kv_pos + shift >= full_len)
                            s = LOWEST;
                        scores[qs][a * 4 + b] = s;
                        local_max = max(local_max, s);
                    }

                const float row_max =
                    reduce_max_groups(local_max)
                    * args.qk_scale_log2;
                const float new_m =
                    __builtin_floorf(max(m_row[qs], row_max));
                if(m_ref[qs] == LOWEST)
                    m_ref[qs] = new_m;
                m_row[qs] = new_m;
                bias[qs] = -m_ref[qs];
                dexp[qs] = (int)(new_m - m_ref[qs]);

                float local_sum = 0.f;
                i32x8 packed;
#pragma unroll
                for(int a = 0; a < T::N_TILES; ++a)
                {
                    vector_t<float, 4> p4;
#pragma unroll
                    for(int b = 0; b < 4; ++b)
                    {
                        const float p = __builtin_amdgcn_exp2f(
                            __builtin_fmaf(
                                scores[qs][a * 4 + b],
                                args.qk_scale_log2,
                                bias[qs]));
                        local_sum += p;
                        p4[b] = __builtin_ldexpf(
                            p, 7 - dexp[qs]);
                    }
                    packed[a] = __builtin_bit_cast(
                        int, cast<fp8_t>(p4));
                }
                v_p[qs] = packed;
                l_row[qs] += reduce_sum_groups(local_sum);
            }

            s_waitcnt_lgkmcnt(0_I);
            auto load_v = [&](auto i_ot) {
                constexpr int ot = decltype(i_ot)::value;
                i32x8 v;
                static_for<T::K_BLOCKS>([&](auto i_beta) {
                    constexpr int beta = i_beta.value;
                    const i32x2 r = __builtin_bit_cast(
                        i32x2,
                        s_kv.template _tr_load<
                            8,
                            ot * T::ATOM
                                + beta * 4 * T::ROW>(pv_base));
                    v[2 * beta] = r[0];
                    v[2 * beta + 1] = r[1];
                });
                return v;
            };

            asm volatile("" ::: "memory");
            s_waitcnt_lgkmcnt(0_I);
            static_for<T::O_TILES / T::PV_BATCH>([&](auto i_batch) {
                constexpr int first =
                    i_batch.value * T::PV_BATCH;
                i32x8 vv[T::PV_BATCH];
                static_for<T::PV_BATCH>([&](auto i) {
                    constexpr int j = i.value;
                    vv[j] = load_v(number<first + j>{});
                });
                s_waitcnt_lgkmcnt(0_I);
#pragma unroll
                for(int j = 0; j < T::PV_BATCH; ++j)
                    asm volatile("" : "+v"(vv[j]) ::);
                static_for<T::PV_BATCH>([&](auto i) {
                    constexpr int j = i.value;
#pragma unroll
                    for(int qs = 0; qs < T::Q_SUB; ++qs)
                        v_o[qs][first + j] = mma_p(
                            vv[j],
                            v_p[qs],
                            v_o[qs][first + j],
                            127 + dexp[qs]);
                });
            });
            if(has_next)
            {
#pragma unroll
                for(int i = 0; i < T::SLOTS_PER_WAVE; ++i)
                    rows0[i] = rows1[i];
            }
            buf ^= 1;
        }
    }

    // The partial accumulator's native C layout scatters one wave over 16
    // head rows.  Reuse the now-dead KV LDS as a transpose buffer so global
    // stores are contiguous.  The barrier keeps an early wave from clobbering
    // the final tile while another wave is still in PV.
    lds_barrier();

    const size_t ml_base =
        ((size_t)req * args.num_splits + split)
        * args.rows_pad;
    if(g == 0)
    {
#pragma unroll
        for(int qs = 0; qs < T::Q_SUB; ++qs)
        {
            const int row =
                h_base + h_wave + qs * T::Q_TILE + c;
            if(row < args.valid_rows)
            {
                args.part_m[ml_base + row] = m_ref[qs];
                args.part_l[ml_base + row] = l_row[qs];
            }
        }
    }

    auto s_ep = make_smem(
        reinterpret_cast<float*>(smem)
        + warp * T::Q_TILE * T::EP_PITCH);
    const size_t acc_base = ml_base * T::D_V;
#pragma unroll
    for(int qs = 0; qs < T::Q_SUB; ++qs)
    {
#pragma unroll
        for(int ot = 0; ot < T::O_TILES; ++ot)
        {
            const f32x4 x = v_o[qs][ot] * 0.0078125f;
            s_ep.template store<4>(
                __builtin_bit_cast(vector_t<float, 4>, x),
                c * T::EP_PITCH + ot * 16 + g * 4);
        }
        s_waitcnt_lgkmcnt(0_I);

#pragma unroll
        for(int hh = 0; hh < T::Q_TILE; ++hh)
        {
            const int row =
                h_base + h_wave + qs * T::Q_TILE + hh;
            if(row < args.valid_rows)
            {
#pragma unroll
                for(int half = 0; half < 2; ++half)
                {
                    const auto x = s_ep.template load<4>(
                        hh * T::EP_PITCH + half * 256 + lane * 4);
                    float* dst =
                        args.part_acc + acc_base
                        + (size_t)row * T::D_V
                        + half * 256 + lane * 4;
                    *reinterpret_cast<vector_t<float, 4>*>(dst) = x;
                }
            }
        }
        s_waitcnt_lgkmcnt(0_I);
    }
}

__global__ __launch_bounds__(KIMI_H40_REDUCE_THREADS, 1)
void reduce_kernel(kargs args)
{
    using device::f32x4;
    const int linear_row = (int)__builtin_amdgcn_workgroup_id_x();
    const int req = linear_row / args.valid_rows;
    const int row = linear_row - req * args.valid_rows;
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    __shared__ float weights[KIMI_H40_REDUCE_THREADS];
    __shared__ float scratch[KIMI_H40_REDUCE_THREADS];

    const size_t req_base =
        (size_t)req * args.num_splits * args.rows_pad;
    float l = 0.f;
    float m = device::LOWEST;
    if(tid < args.num_splits)
    {
        const size_t off =
            req_base + (size_t)tid * args.rows_pad + row;
        l = args.part_l[off];
        if(l > 0.f)
            m = args.part_m[off];
    }
    scratch[tid] = m;
    __builtin_amdgcn_s_barrier();
    for(int stride = KIMI_H40_REDUCE_THREADS / 2;
        stride > 0;
        stride >>= 1)
    {
        if(tid < stride)
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        __builtin_amdgcn_s_barrier();
    }
    const float global_m = scratch[0];

    float w = 0.f;
    if(tid < args.num_splits && l > 0.f)
        w = __builtin_amdgcn_exp2f(m - global_m);
    weights[tid] = w;
    scratch[tid] = w * l;
    __builtin_amdgcn_s_barrier();
    for(int stride = KIMI_H40_REDUCE_THREADS / 2;
        stride > 0;
        stride >>= 1)
    {
        if(tid < stride)
            scratch[tid] += scratch[tid + stride];
        __builtin_amdgcn_s_barrier();
    }
    const float denom = scratch[0];

    auto* out = reinterpret_cast<__bf16*>(args.out_ptr);
    const size_t req_acc =
        (size_t)req * args.num_splits
        * args.rows_pad * traits::D_V;
    for(int col = tid * 4;
        col < traits::D_V;
        col += KIMI_H40_REDUCE_THREADS * 4)
    {
        f32x4 acc{0.f, 0.f, 0.f, 0.f};
        for(int s = 0; s < args.num_splits; ++s)
        {
            const size_t off =
                req_acc
                + ((size_t)s * args.rows_pad + row) * traits::D_V
                + col;
            const f32x4 x =
                *reinterpret_cast<const f32x4*>(args.part_acc + off);
            acc += x * weights[s];
        }
        const float scale =
            denom > 0.f ? args.output_scale / denom : 0.f;
        const f32x4 y = acc * scale;
        __bf16* dst =
            out + ((size_t)req * args.valid_rows + row) * traits::D_V + col;
#pragma unroll
        for(int i = 0; i < 4; ++i)
            dst[i] = (__bf16)y[i];
    }
}

#else

__global__ void partial_kernel(kargs) {}
__global__ void reduce_kernel(kargs) {}

#endif

} // namespace kimi_verify_h40

#ifndef __HIP_DEVICE_COMPILE__

extern "C" int kimi_verify_h40_run(
    const void* q,
    const void* kv,
    const int* kv_indptr,
    const int* kv_indices,
    float* part_m,
    float* part_l,
    float* part_acc,
    void* out,
    int num_requests,
    int qlen,
    int nhead,
    int kv_rows,
    int num_splits,
    float qk_scale_log2,
    float output_scale,
    void* stream_ptr)
{
    using namespace kimi_verify_h40;
    kargs args{};
    args.q_ptr = q;
    args.kv_ptr = kv;
    args.kv_indptr = kv_indptr;
    args.kv_indices = kv_indices;
    args.part_m = part_m;
    args.part_l = part_l;
    args.part_acc = part_acc;
    args.out_ptr = out;
    args.T = num_requests;
    args.qlen = qlen;
    args.nhead = nhead;
    args.valid_rows = qlen * nhead;
    args.rows_pad = args.valid_rows;
    args.kv_rows = kv_rows;
    args.num_splits = num_splits;
    args.qk_scale_log2 = qk_scale_log2;
    args.output_scale = output_scale;

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    const dim3 grid(
        num_requests,
        ceil_div(args.valid_rows, traits::HEADS_PER_BLOCK),
        num_splits);
    hipLaunchKernelGGL(
        partial_kernel,
        grid,
        dim3(traits::BLOCK_SIZE),
        0,
        stream,
        args);
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
        return (int)err;

    hipLaunchKernelGGL(
        reduce_kernel,
        dim3(num_requests * args.valid_rows),
        dim3(KIMI_H40_REDUCE_THREADS),
        0,
        stream,
        args);
    return (int)hipGetLastError();
}

extern "C" int kimi_verify_h40_partial(
    const void* q,
    const void* kv,
    const int* kv_indptr,
    const int* kv_indices,
    float* part_m,
    float* part_l,
    float* part_acc,
    int num_requests,
    int qlen,
    int nhead,
    int kv_rows,
    int num_splits,
    float qk_scale_log2,
    void* stream_ptr)
{
    using namespace kimi_verify_h40;
    kargs args{};
    args.q_ptr = q;
    args.kv_ptr = kv;
    args.kv_indptr = kv_indptr;
    args.kv_indices = kv_indices;
    args.part_m = part_m;
    args.part_l = part_l;
    args.part_acc = part_acc;
    args.T = num_requests;
    args.qlen = qlen;
    args.nhead = nhead;
    args.valid_rows = qlen * nhead;
    args.rows_pad = args.valid_rows;
    args.kv_rows = kv_rows;
    args.num_splits = num_splits;
    args.qk_scale_log2 = qk_scale_log2;

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    hipLaunchKernelGGL(
        partial_kernel,
        dim3(
            num_requests,
            ceil_div(args.valid_rows, traits::HEADS_PER_BLOCK),
            num_splits),
        dim3(traits::BLOCK_SIZE),
        0,
        stream,
        args);
    return (int)hipGetLastError();
}

extern "C" int kimi_verify_h40_reduce(
    const float* part_m,
    const float* part_l,
    const float* part_acc,
    void* out,
    int num_requests,
    int qlen,
    int nhead,
    int num_splits,
    float output_scale,
    void* stream_ptr)
{
    using namespace kimi_verify_h40;
    kargs args{};
    args.part_m = const_cast<float*>(part_m);
    args.part_l = const_cast<float*>(part_l);
    args.part_acc = const_cast<float*>(part_acc);
    args.out_ptr = out;
    args.T = num_requests;
    args.qlen = qlen;
    args.nhead = nhead;
    args.valid_rows = qlen * nhead;
    args.rows_pad = args.valid_rows;
    args.num_splits = num_splits;
    args.output_scale = output_scale;

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    hipLaunchKernelGGL(
        reduce_kernel,
        dim3(num_requests * args.valid_rows),
        dim3(KIMI_H40_REDUCE_THREADS),
        0,
        stream,
        args);
    return (int)hipGetLastError();
}

extern "C" int kimi_verify_h40_partial_range(
    const void* q,
    const void* kv,
    const int* kv_indptr,
    const int* kv_indices,
    float* part_m,
    float* part_l,
    float* part_acc,
    int num_requests,
    int qlen,
    int nhead,
    int kv_rows,
    int num_splits,
    float qk_scale_log2,
    int h_offset,
    int h_blocks,
    int rows_pad,
    void* stream_ptr)
{
    using namespace kimi_verify_h40;
    kargs args{};
    args.q_ptr = q;
    args.kv_ptr = kv;
    args.kv_indptr = kv_indptr;
    args.kv_indices = kv_indices;
    args.part_m = part_m;
    args.part_l = part_l;
    args.part_acc = part_acc;
    args.T = num_requests;
    args.qlen = qlen;
    args.nhead = nhead;
    args.valid_rows = qlen * nhead;
    args.rows_pad = rows_pad;
    args.h_offset = h_offset;
    args.kv_rows = kv_rows;
    args.num_splits = num_splits;
    args.qk_scale_log2 = qk_scale_log2;

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    hipLaunchKernelGGL(
        partial_kernel,
        dim3(num_requests, h_blocks, num_splits),
        dim3(traits::BLOCK_SIZE),
        0,
        stream,
        args);
    return (int)hipGetLastError();
}

#endif

