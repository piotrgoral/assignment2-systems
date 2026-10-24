"""FlashAttention-2 as torch.autograd.Function subclasses.

FlashAttentionPytorch is a pure-PyTorch (no Triton) version of the tiled forward pass (Algorithm 1 in the handout):
queries are split into tiles of Q_TILE_SIZE and keys/values into tiles of K_TILE_SIZE, and the softmax is computed
online with a running row max m and a running denominator l, so the full N_q x N_k attention matrix is never
materialized. It is slow, but it gives us a reference to debug the Triton kernel against.

The forward pass saves L (logsumexp of the attention scores), Q, K, V and O for the backward pass.
"""

import math

import torch


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
        raise NotImplementedError
