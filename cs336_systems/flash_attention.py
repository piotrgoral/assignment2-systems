"""FlashAttention-2 as torch.autograd.Function subclasses.

FlashAttentionPytorch is a pure-PyTorch (no Triton) version of the tiled forward pass (Algorithm 1 in the handout):
queries are split into tiles of Q_TILE_SIZE and keys/values into tiles of K_TILE_SIZE, and the softmax is computed
online with a running row max m and a running denominator l, so the full N_q x N_k attention matrix is never
materialized. It is slow, but it gives us a reference to debug the Triton kernel against.

FlashAttentionTriton runs the same algorithm as one fused Triton kernel (flash_fwd_kernel). The launch grid is
(T_q, batch_size): each program instance owns one query tile of one batch element, keeps Q_i and the fp32 state
(O_i, l, m) on chip and loops once over the key tiles. Leading batch dims are flattened into one before the launch.

With is_causal=True, both forward passes add -1e6 to the scores of keys that come after the query. The Triton kernel
also stops its key loop at the diagonal: later key tiles are fully masked and would add exactly 0.

The Triton kernel takes any sequence lengths: out-of-range keys of the last tile are loaded as zeros and their scores
set to -inf (as in the lecture's fused softmax), and out-of-range query rows are never stored. With fp32 inputs both
tl.dot calls use input_precision="ieee" (true fp32, like the PyTorch baseline) instead of Triton's TF32 default.

FlashAttentionTriton picks its tile sizes (and num_warps / num_stages) with triton.autotune, per sequence lengths,
head dim, dtype and is_causal, unless Q_TILE_SIZE / K_TILE_SIZE are set on the class, which forces fixed tiles.

Both forward passes save L (logsumexp of the attention scores), Q, K, V and O for the backward pass. The backward
(shared by both classes) is plain PyTorch under torch.compile: it recomputes P = exp(S - L) from Q, K and L instead
of reading a stored attention matrix, and uses D = rowsum(O * dO) so no softmax is needed (Eqs. 13-19).
"""

import math

import torch
import triton
import triton.language as tl


def _causal_mask(n_queries, n_keys, device):
    return torch.arange(n_queries, device=device)[:, None] >= torch.arange(n_keys, device=device)[None, :]


def _flash_backward(Q, K, V, O, dO, L, is_causal: bool):
    scale = 1 / math.sqrt(Q.shape[-1])
    Qf, Kf, Vf, Of, dOf = (t.float() for t in (Q, K, V, O, dO))

    D = (Of * dOf).sum(dim=-1)
    S = Qf @ Kf.mT * scale  # (13)
    if is_causal:
        S = torch.where(_causal_mask(Q.shape[-2], K.shape[-2], Q.device), S, S - 1e6)
    P = torch.exp(S - L[..., None])  # (14)
    dV = P.mT @ dOf  # (15)
    dP = dOf @ Vf.mT  # (16)
    dS = P * (dP - D[..., None])  # (17)
    dQ = dS @ Kf * scale  # (18)
    dK = dS.mT @ Qf * scale  # (19)
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


_flash_backward_compiled = torch.compile(_flash_backward)


class FlashAttentionPytorch(torch.autograd.Function):
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        n_queries, d = Q.shape[-2:]
        n_keys = K.shape[-2]
        Bq, Bk = FlashAttentionPytorch.Q_TILE_SIZE, FlashAttentionPytorch.K_TILE_SIZE
        assert n_queries % Bq == 0 and n_keys % Bk == 0, "sequence lengths must be multiples of the tile sizes"
        scale = 1 / math.sqrt(d)
        batch_shape = Q.shape[:-2]

        O = torch.empty_like(Q)
        L = torch.empty(*batch_shape, n_queries, device=Q.device, dtype=torch.float32)

        for q_start in range(0, n_queries, Bq):
            Q_i = Q[..., q_start : q_start + Bq, :].float()
            O_i = torch.zeros(*batch_shape, Bq, d, device=Q.device, dtype=torch.float32)
            l_i = torch.zeros(*batch_shape, Bq, device=Q.device, dtype=torch.float32)
            m_i = torch.full((*batch_shape, Bq), -torch.inf, device=Q.device, dtype=torch.float32)

            for k_start in range(0, n_keys, Bk):
                K_j = K[..., k_start : k_start + Bk, :].float()
                V_j = V[..., k_start : k_start + Bk, :].float()

                S_ij = Q_i @ K_j.mT * scale
                if is_causal:
                    mask = _causal_mask(n_queries, n_keys, Q.device)[q_start : q_start + Bq, k_start : k_start + Bk]
                    S_ij = torch.where(mask, S_ij, S_ij - 1e6)
                m_new = torch.maximum(m_i, S_ij.amax(dim=-1))
                P_ij = torch.exp(S_ij - m_new[..., None])
                alpha = torch.exp(m_i - m_new)
                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha[..., None] * O_i + P_ij @ V_j
                m_i = m_new

            O[..., q_start : q_start + Bq, :] = (O_i / l_i[..., None]).to(Q.dtype)
            L[..., q_start : q_start + Bq] = m_i + torch.log(l_i)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, grad_out, L, ctx.is_causal)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):  # fmt: skip
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index multiplied with the batch stride for each tensor.
    # Q, O and L point at this program's query tile; K and V start at the first key tile and advance in the loop.
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # On-chip online softmax state, in fp32 whatever the input dtype.
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        # Key tiles past the last query of this tile are fully masked.
        n_key_tiles = tl.minimum(n_key_tiles, tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE))
    for j in range(n_key_tiles):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j), input_precision=INPUT_PRECISION) * scale
        k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        if is_causal:
            S_ij = tl.where(q_idx[:, None] >= k_idx[None, :], S_ij, S_ij - 1e6)
        # Padded keys past N_KEYS must not count in the softmax (every tile has at least one real key).
        S_ij = tl.where(k_idx[None, :] < N_KEYS, S_ij, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(P_ij, axis=1)
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_i * alpha[:, None], input_precision=INPUT_PRECISION)
        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


_TILE_CONFIGS = [
    triton.Config({"Q_TILE_SIZE": bq, "K_TILE_SIZE": bk}, num_warps=num_warps, num_stages=num_stages)
    for bq, bk in [(16, 16), (32, 32), (32, 64), (64, 32), (64, 64), (128, 32), (128, 64), (128, 128)]
    for num_warps in (4, 8)
    for num_stages in (2, 3)
]

flash_fwd_kernel_autotuned = triton.autotune(
    configs=_TILE_CONFIGS,
    key=["N_QUERIES", "N_KEYS", "D", "is_causal"],  # the autotuner also keys on the tensors' dtypes
)(flash_fwd_kernel)


class FlashAttentionTriton(torch.autograd.Function):
    # None: autotune the tiles. Set both to an int to force fixed square or rectangular tiles.
    Q_TILE_SIZE = None
    K_TILE_SIZE = None

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernels need CUDA tensors"
        n_queries, d = Q.shape[-2:]
        n_keys = K.shape[-2]
        Bq, Bk = FlashAttentionTriton.Q_TILE_SIZE, FlashAttentionTriton.K_TILE_SIZE
        autotune = Bq is None
        batch_shape = Q.shape[:-2]

        # The grid has a single batch axis, so flatten all leading dims into one.
        Q3, K3, V3 = Q.reshape(-1, n_queries, d), K.reshape(-1, n_keys, d), V.reshape(-1, n_keys, d)
        batch_size = Q3.shape[0]
        O = torch.empty_like(Q3)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=torch.float32)

        if autotune:
            kernel = flash_fwd_kernel_autotuned
            grid = lambda meta: (triton.cdiv(n_queries, meta["Q_TILE_SIZE"]), batch_size)  # noqa: E731
            tiles = {}
        else:
            kernel = flash_fwd_kernel
            grid = (triton.cdiv(n_queries, Bq), batch_size)
            tiles = dict(Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk)
        kernel[grid](
            Q3, K3, V3,
            O, L,
            Q3.stride(0), Q3.stride(1), Q3.stride(2),
            K3.stride(0), K3.stride(1), K3.stride(2),
            V3.stride(0), V3.stride(1), V3.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            1 / math.sqrt(d),
            D=d,
            is_causal=is_causal,
            INPUT_PRECISION="ieee" if Q.dtype == torch.float32 else "tf32",
            **tiles,
        )  # fmt: skip

        O = O.view(Q.shape)
        L = L.view(*batch_shape, n_queries)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, grad_out, L, ctx.is_causal)
        return dQ, dK, dV, None
