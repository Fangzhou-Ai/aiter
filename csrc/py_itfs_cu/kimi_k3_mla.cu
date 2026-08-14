// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Prebuilt-code-object entry points for the Kimi-K3 DSpark MLA decode kernels.
//
// The kernels themselves are compiled ahead of time into
// hsa/gfx950/kimi_k3_mla/{draft,verify}_mla.co (see the commit message for the
// exact build), so nothing here
// is JIT-compiled and the deployment image needs no compiler.  What this file
// owns is the half that cannot live in a code object: the launch policy (row
// tile, split count, grid order) and the argument buffer.
//
// Draft and verify share every line of that policy -- they are the same fold
// with opposite masks -- so they share this file and differ only in which .co
// and which wide-path symbol they name.

#include "aiter_hip_common.h"
#include "aiter_tensor.h"

#include <cstddef>

namespace {

// Mirrors kimi_k3_draft::kargs and kimi_k3_verify::kargs, which are the same
// layout by construction.  Deliberately NOT packed: the device struct
// is naturally aligned, and packing here would shift every field after the
// first int pair.  The static_assert is the guard -- a layout change on either
// side turns a silent wrong-answer into a build failure.
struct KimiK3MlaArgs
{
    const void* q_ptr;
    const void* kv_ptr;
    const int* kv_indptr;
    const int* kv_indices;
    const int* kv_block_table;
    const int* kv_seq_lens;
    int32_t bt_stride;
    int32_t page_size;
    float* part_m;
    float* part_l;
    float* part_acc;
    void* out_ptr;
    int32_t T;
    int32_t qlen;
    int32_t nhead;
    int32_t valid_rows;
    int32_t rows_pad;
    int32_t h_offset;
    int32_t kv_rows;
    int32_t num_splits;
    float qk_scale_log2;
    float output_scale;
};
static_assert(sizeof(KimiK3MlaArgs) == 128, "kargs layout drifted from the .co");
static_assert(offsetof(KimiK3MlaArgs, part_m) == 56, "kargs layout drifted");
static_assert(offsetof(KimiK3MlaArgs, T) == 88, "kargs layout drifted");

// A 32-row MFMA tile cannot express the 64-row narrow block, so rows <= 64 keep
// the 16x16x128 kernel on both paths.
constexpr int NARROW_ROWS = 64;
constexpr int WIDE_ROWS = 128;
constexpr int BLOCK_SIZE = 256;      // NUM_WARPS 4 * WARP_SIZE 64
constexpr int REDUCE_THREADS = 128;
constexpr int KV_TILE = 128;

// The two code objects carry separate namespaces -- kimi_k3_draft and
// kimi_k3_verify -- so a symbol names exactly one kernel and cannot be resolved
// against the wrong .co. Keep it that way: the two disagree on grid axis order
// (verify is (T, splits, row_blocks), draft is (T, row_blocks, splits)), so a
// mismatched pair computes the wrong rows and still returns finite numbers.
constexpr const char* SYM_V_NARROW = "_ZN14kimi_k3_verify14partial_kernelILi1EEEvNS_5kargsE";
constexpr const char* SYM_V_W32 = "_ZN14kimi_k3_verify18partial_kernel_w32ENS_5kargsE";
constexpr const char* SYM_V_REDUCE = "_ZN14kimi_k3_verify13reduce_kernelENS_5kargsE";

constexpr const char* SYM_D_NARROW = "_ZN13kimi_k3_draft14partial_kernelILi1EEEvNS_5kargsE";
constexpr const char* SYM_D_QSUB2 = "_ZN13kimi_k3_draft14partial_kernelILi2EEEvNS_5kargsE";
constexpr const char* SYM_D_REDUCE = "_ZN13kimi_k3_draft13reduce_kernelENS_5kargsE";

int fill_args(KimiK3MlaArgs& args,
              aiter_tensor_t* q,
              aiter_tensor_t* kv,
              aiter_tensor_t* out,
              aiter_tensor_t* block_table,
              aiter_tensor_t* seq_lens,
              aiter_tensor_t* part_m,
              aiter_tensor_t* part_l,
              aiter_tensor_t* part_acc,
              int64_t qlen,
              int64_t num_splits,
              float qk_scale_log2,
              float output_scale)
{
    if(q == nullptr || kv == nullptr || out == nullptr || block_table == nullptr ||
       seq_lens == nullptr || part_m == nullptr || part_l == nullptr || part_acc == nullptr)
        return -20;
    if(q->dtype() != AITER_DTYPE_fp8 || kv->dtype() != AITER_DTYPE_fp8)
        return -21;
    if(out->dtype() != AITER_DTYPE_bf16)
        return -22;
    if(block_table->dtype() != AITER_DTYPE_i32 || seq_lens->dtype() != AITER_DTYPE_i32)
        return -23;
    if(part_m->dtype() != AITER_DTYPE_fp32 || part_l->dtype() != AITER_DTYPE_fp32 ||
       part_acc->dtype() != AITER_DTYPE_fp32)
        return -24;
    if(q->dim() != 3 || kv->dim() < 3 || out->dim() != 3 || block_table->dim() != 2)
        return -25;
    if(qlen < 1 || num_splits < 1 || qlen > (1 << 20))
        return -26;

    const int nhead = static_cast<int>(q->size(1));
    const int T = static_cast<int>(seq_lens->numel());
    if(T < 1 || static_cast<int>(q->size(0)) != T * static_cast<int>(qlen))
        return -27;
    if(block_table->size(0) < T)
        return -28;
    // The kernel takes a row stride, so an over-wide or row-sliced table is
    // fine, but the innermost stride must be 1 -- it indexes pages directly.
    if(block_table->stride(1) != 1)
        return -29;

    const int page_size = static_cast<int>(kv->size(1));
    if(page_size % KV_TILE)
        return -2; // a KV tile would straddle a page

    args = KimiK3MlaArgs{};
    args.q_ptr = q->data_ptr();
    args.kv_ptr = kv->data_ptr();
    args.kv_block_table = static_cast<const int*>(block_table->data_ptr());
    args.kv_seq_lens = static_cast<const int*>(seq_lens->data_ptr());
    args.bt_stride = static_cast<int32_t>(block_table->stride(0));
    args.page_size = static_cast<int32_t>(page_size);
    args.part_m = static_cast<float*>(part_m->data_ptr());
    args.part_l = static_cast<float*>(part_l->data_ptr());
    args.part_acc = static_cast<float*>(part_acc->data_ptr());
    args.out_ptr = out->data_ptr();
    args.T = T;
    args.qlen = static_cast<int32_t>(qlen);
    args.nhead = nhead;
    args.valid_rows = static_cast<int32_t>(qlen) * nhead;
    args.rows_pad = args.valid_rows;
    args.kv_rows = static_cast<int32_t>(kv->size(0)) * page_size;
    args.num_splits = static_cast<int32_t>(num_splits);
    args.qk_scale_log2 = qk_scale_log2;
    args.output_scale = output_scale;
    return 0;
}

// Workgroups go to the 8 XCDs round-robin by linear id
// (id = x + gridDim.x*(y + gridDim.y*z)), and the row-blocks of one
// (request, split) read the IDENTICAL KV segment -- whether they land on the
// same XCD decides whether the second read is served by that XCD's L2.  With
// the row-block on z their ids differ by T*num_splits, which the launch rule
// pins at ~256 and is always a multiple of 8; on y they would differ by T.
// zswap: verify orders the grid (T, splits, row_blocks); draft is the earlier
// (T, row_blocks, splits).  This is NOT cosmetic -- the kernel reads its row
// block off a fixed axis, so the wrong order silently computes the wrong rows.
void launch(AiterAsmKernel& partial,
            AiterAsmKernel& reduce,
            KimiK3MlaArgs& args,
            bool zswap,
            hipStream_t stream)
{
    size_t arg_size = sizeof(args);
    const bool narrow = args.valid_rows <= NARROW_ROWS;
    const int hpb = narrow ? NARROW_ROWS : WIDE_ROWS;
    const int row_blocks = (args.valid_rows + hpb - 1) / hpb;

    const int gy = zswap ? args.num_splits : row_blocks;
    const int gz = zswap ? row_blocks : args.num_splits;
    partial.launch_kernel({&args, &arg_size,
                           args.T, gy, gz,
                           BLOCK_SIZE, 1, 1,
                           stream});
    reduce.launch_kernel({&args, &arg_size,
                          args.T * args.valid_rows, 1, 1,
                          REDUCE_THREADS, 1, 1,
                          stream});
}

// One instance per (symbol, code object).  Construction is what registers, and
// registering is illegal while a stream is capturing, so warmup() and the decode
// entries must share these -- give each entry its own function-local static and
// the warmup registers a second, unused set while the decode path still
// constructs on first call, inside the capture.
AiterAsmKernel& k_verify_narrow()
{ static AiterAsmKernel k(SYM_V_NARROW, "/kimi_k3_mla/verify_mla.co"); return k; }
AiterAsmKernel& k_verify_wide()
{ static AiterAsmKernel k(SYM_V_W32, "/kimi_k3_mla/verify_mla.co"); return k; }
AiterAsmKernel& k_verify_reduce()
{ static AiterAsmKernel k(SYM_V_REDUCE, "/kimi_k3_mla/verify_mla.co"); return k; }
AiterAsmKernel& k_draft_narrow()
{ static AiterAsmKernel k(SYM_D_NARROW, "/kimi_k3_mla/draft_mla.co"); return k; }
AiterAsmKernel& k_draft_wide()
{ static AiterAsmKernel k(SYM_D_QSUB2, "/kimi_k3_mla/draft_mla.co"); return k; }
AiterAsmKernel& k_draft_reduce()
{ static AiterAsmKernel k(SYM_D_REDUCE, "/kimi_k3_mla/draft_mla.co"); return k; }

} // namespace

// Registering a code object is illegal while a stream is capturing, and the
// first call can land inside cudagraph capture -- vLLM's pre-capture eager pass
// is itself skippable (VLLM_SKIP_KERNEL_WARMUP).  So registration is forced
// here, from a call the Python side makes at import time.  Construction is what
// registers; nothing is launched.
AITER_C_ITFS int kimi_k3_mla_warmup()
{
    k_verify_narrow(); k_verify_wide(); k_verify_reduce();
    k_draft_narrow();  k_draft_wide();  k_draft_reduce();
    return 0;
}

AITER_C_ITFS int kimi_k3_verify_mla_decode(aiter_tensor_t* q,
                                               aiter_tensor_t* kv,
                                               aiter_tensor_t* out,
                                               aiter_tensor_t* block_table,
                                               aiter_tensor_t* seq_lens,
                                               aiter_tensor_t* part_m,
                                               aiter_tensor_t* part_l,
                                               aiter_tensor_t* part_acc,
                                               int64_t qlen,
                                               int64_t num_splits,
                                               float qk_scale_log2,
                                               float output_scale,
                                               void* stream)
{
    KimiK3MlaArgs args;
    const int err = fill_args(args, q, kv, out, block_table, seq_lens,
                              part_m, part_l, part_acc, qlen, num_splits,
                              qk_scale_log2, output_scale);
    if(err)
        return err;

    const HipDeviceGuard device_guard(q->device_id);
    const bool narrow = args.valid_rows <= NARROW_ROWS;

    launch(narrow ? k_verify_narrow() : k_verify_wide(), k_verify_reduce(),
           args, /*zswap=*/true,
           static_cast<hipStream_t>(stream));
    return 0;
}

AITER_C_ITFS int dspark_draft_mla_decode(aiter_tensor_t* q,
                                              aiter_tensor_t* kv,
                                              aiter_tensor_t* out,
                                              aiter_tensor_t* block_table,
                                              aiter_tensor_t* seq_lens,
                                              aiter_tensor_t* part_m,
                                              aiter_tensor_t* part_l,
                                              aiter_tensor_t* part_acc,
                                              int64_t qlen,
                                              int64_t num_splits,
                                              float qk_scale_log2,
                                              float output_scale,
                                              void* stream)
{
    KimiK3MlaArgs args;
    const int err = fill_args(args, q, kv, out, block_table, seq_lens,
                              part_m, part_l, part_acc, qlen, num_splits,
                              qk_scale_log2, output_scale);
    if(err)
        return err;

    const HipDeviceGuard device_guard(q->device_id);
    const bool narrow = args.valid_rows <= NARROW_ROWS;

    // The draft build is KIMI_H40_W32=0, so its wide path is partial_kernel<2>
    // and draft_mla.co carries no partial_kernel_w32 symbol at all.  Naming the
    // verify symbol here would fail at load, not at build.
    launch(narrow ? k_draft_narrow() : k_draft_wide(), k_draft_reduce(),
           args, /*zswap=*/false,
           static_cast<hipStream_t>(stream));
    return 0;
}
