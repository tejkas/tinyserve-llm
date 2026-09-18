"""Naive PyTorch paged attention. The correctness oracle for flash-attn backend.."""

from __future__ import annotations

import torch


def paged_attention_reference(
        q: torch.Tensor,    # [Batch, Num_Heads, Head_Dim]
        k_cache: torch.Tensor, # [num_blocks, block_size, num_heads, head_dim]
        v_cache: torch.Tensor, # [num_blocks, block_size, num_heads, head_dim]
        block_tables: torch.Tensor, # [batch, max_blocks_per_seq]
        seq_lens: torch.Tensor, # [batch]
        block_size: int
) -> torch.Tensor:  # [batch, num_heads, head_dim]
    batch_size, num_heads, head_dim = q.shape
    output = torch.zeros_like(q)

    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id].item() 
        # ceiling division
        num_blocks = (seq_len + block_size - 1) // block_size

        k_blocks = []
        v_blocks = []
        for block in range(num_blocks):
            # collect k,v blocks from block tables
            block_id = block_tables[batch_id, block].item()
            k_blocks.append(k_cache[block_id])
            v_blocks.append(v_cache[block_id])
        
        # concat kv-blocks into one tensor, then grab first `seq_len` tokens
        # results in a clean k, v tensor set for attention math
        k = torch.cat(k_blocks, dim=0)[:seq_len] # [seq_len, num_heads, head_dim]
        v = torch.cat(v_blocks, dim=0)[:seq_len] # [seq_len, num_heads, head_dim]

        scale = head_dim ** -0.5 # 1/sqrt(d)

        for h in range(num_heads):
            q_vec = q[batch_id, h] # [head_dim]
            k_mat = k[:, h, :] # [seq_len, head_dim]
            v_mat = v[:, h, :] # [seq_len, head_dim]

            scores = k_mat @ q_vec # [seq_len]
            scores = scores * scale
            weights = torch.softmax(scores, dim=0) # [seq_len]
            output[batch_id, h] = weights @ v_mat # [head_dim]
    
    return output
