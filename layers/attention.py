import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Custom multi-head attention mechanism that allows adding custom bias terms to attention scores

    Args:
        embed_dim: Input feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
        bias: Whether to use bias in linear projection layers
        batch_first: Whether input format is [batch, seq, features]
    """

    def __init__(self,
                 dim_q,
                 dim_k,
                 dim_v,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 mc_drop=False,
                 bias=True,
                 batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mc_drop = mc_drop
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim**-0.5

        # Add LayerNorm layers before linear projections
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_k)
        self.norm3 = nn.LayerNorm(dim_v)

        # Create linear projection layers for Q, K, V
        self.q_proj = nn.Linear(dim_q, embed_dim, bias=bias)
        self.k_proj = nn.Linear(dim_k, embed_dim, bias=bias)
        self.v_proj = nn.Linear(dim_v, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize projection layer parameters
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                return_attn_scores=False,
                value_mask=None):
        """
        Forward pass

        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_v, embed_dim]
            key_padding_mask: Key padding mask [batch_size, seq_len_k]
            attn_mask: Attention mask [seq_len_q, seq_len_k] or [batch_size, num_heads, seq_len_q, seq_len_k]
            return_attn_scores: Whether to return attention scores (before softmax)

        Returns:
            output: Output tensor [batch_size, seq_len_q, embed_dim]
            attn_scores: Attention scores (if return_attn_scores=True)
        """
        if not self.batch_first:
            # Convert to batch_first format
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # Apply LayerNorm before projections
        query = self.norm1(query)
        key = self.norm2(key)
        value = self.norm3(value)

        # Project to multi-head attention space
        q = self.q_proj(query) * self.scaling  # [batch_size, tgt_len, embed_dim]
        k = self.k_proj(key)  # [batch_size, src_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, src_len, embed_dim] #FIXME Normalization precedes projection.

        # Split feature dimension into multiple heads
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(
            1, 2)  # [batch_size, num_heads, tgt_len, head_dim]
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
            1, 2)  # [batch_size, num_heads, src_len, head_dim]
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
            1, 2)  # [batch_size, num_heads, src_len, head_dim]

        # Calculate attention scores: Q * K^T
        attn_scores = torch.matmul(q, k.transpose(
            -2, -1))  # [batch_size, num_heads, tgt_len, src_len]

        # Apply attention mask (if provided)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [1, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch_size, tgt_len, src_len] -> [batch_size, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(1)

            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))

        if value_mask is not None:
            invalid = value_mask.squeeze(-1).unsqueeze(1).unsqueeze(2)
            all_invalid = invalid.all(dim=-1)
            invalid[all_invalid] = False
            attn_scores = attn_scores.masked_fill(invalid, -float('inf'))

        # Apply key padding mask (if provided)
        if key_padding_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float('inf'))

        # Calculate attention weights
        attn_weights = F.softmax(attn_scores,
                                 dim=-1)  # [batch_size, num_heads, tgt_len, src_len]
        if self.mc_drop:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
        else:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Calculate output: attention * V
        output = torch.matmul(attn_weights,
                              v)  # [batch_size, num_heads, tgt_len, head_dim]

        # Combine multi-head results
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len,
                                                          self.embed_dim)

        # Final linear projection
        output = self.out_proj(output)

        if not self.batch_first:
            # Convert back
            output = output.transpose(0, 1)
        if return_attn_scores:
            return output, attn_scores
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}('\
            f'embed_dim={self.embed_dim}, '\
            f'num_heads={self.num_heads}, '\
            f'dropout={self.dropout}, '\
            f'mc_drop={self.mc_drop}, '\
            f'batch_first={self.batch_first})'


class MultiHeadQueryAttention(nn.Module):

    def __init__(self,
                 dim_q,
                 dim_k,
                 dim_v,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 mc_drop=False,
                 bias=True,
                 batch_first=True,
                 num_query=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mc_drop = mc_drop
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim**-0.5

        # Add LayerNorm layers before linear projections
        self.norm_qs = nn.ModuleList([nn.LayerNorm(dim_q) for _ in range(num_query)])
        self.norm_k = nn.LayerNorm(dim_k)
        self.norm_v = nn.LayerNorm(dim_v)

        # Create linear projection layers for Q, K, V
        self.q_projs = nn.ModuleList(
            nn.Linear(dim_q, embed_dim, bias=bias) for _ in range(num_query))
        self.k_proj = nn.Linear(dim_k, embed_dim, bias=bias)
        self.v_proj = nn.Linear(dim_v, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize projection layer parameters
        for q_proj in self.q_projs:
            nn.init.xavier_uniform_(q_proj.weight)
            if q_proj.bias is not None:
                nn.init.constant_(q_proj.bias, 0.)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                return_attn_scores=False,
                value_mask=None,
                query_type_indices=None):
        """
        Forward pass

        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_v, embed_dim]
            key_padding_mask: Key padding mask [batch_size, seq_len_k]
            attn_mask: Attention mask [seq_len_q, seq_len_k] or [batch_size, num_heads, seq_len_q, seq_len_k]
            return_attn_scores: Whether to return attention scores (before softmax)

        Returns:
            output: Output tensor [batch_size, seq_len_q, embed_dim]
            attn_scores: Attention scores (if return_attn_scores=True)
        """
        if not self.batch_first:
            # Convert to batch_first format
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # Apply LayerNorm before projections
        if query_type_indices is not None:
            query_list = []
            for i in range(len(self.norm_qs)):
                q_subset = query[query_type_indices == i].view(batch_size, -1,
                                                               query.size(-1))
                q_subset = self.norm_qs[i](q_subset)
                q_subset = self.q_projs[i](q_subset) * self.scaling
                query_list.append(q_subset)
            q = torch.zeros(batch_size, tgt_len, self.embed_dim).to(query.device)
            for i in range(len(self.norm_qs)):
                q[query_type_indices == i] = query_list[i].view(-1, self.embed_dim)

        else:
            q = self.norm_qs[0](query)
            q = self.q_projs[0](query) * self.scaling

        key = self.norm_k(key)
        value = self.norm_v(value)
        k = self.k_proj(key)  # [batch_size, src_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, src_len, embed_dim]

        # Split feature dimension into multiple heads
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(
            1, 2)  # [batch_size, num_heads, tgt_len, head_dim]
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
            1, 2)  # [batch_size, num_heads, src_len, head_dim]
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
            1, 2)  # [batch_size, num_heads, src_len, head_dim]

        # Calculate attention scores: Q * K^T
        attn_scores = torch.matmul(q, k.transpose(
            -2, -1))  # [batch_size, num_heads, tgt_len, src_len]

        # Apply attention mask (if provided)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [1, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch_size, tgt_len, src_len] -> [batch_size, 1, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(1)

            attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))

        if value_mask is not None:
            invalid = value_mask.squeeze(-1).unsqueeze(1).unsqueeze(2)
            all_invalid = invalid.all(dim=-1)
            invalid[all_invalid] = False
            attn_scores = attn_scores.masked_fill(invalid, -float('inf'))

        # Apply key padding mask (if provided)
        if key_padding_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float('inf'))

        # Calculate attention weights
        attn_weights = F.softmax(attn_scores,
                                 dim=-1)  # [batch_size, num_heads, tgt_len, src_len]
        if self.mc_drop:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
        else:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Calculate output: attention * V
        output = torch.matmul(attn_weights,
                              v)  # [batch_size, num_heads, tgt_len, head_dim]

        # Combine multi-head results
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len,
                                                          self.embed_dim)

        # Final linear projection
        output = self.out_proj(output)

        if not self.batch_first:
            # Convert back
            output = output.transpose(0, 1)
        if return_attn_scores:
            return output, attn_scores
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}('\
            f'embed_dim={self.embed_dim}, '\
            f'num_heads={self.num_heads}, '\
            f'dropout={self.dropout}, '\
            f'mc_drop={self.mc_drop}, '\
            f'batch_first={self.batch_first})'
