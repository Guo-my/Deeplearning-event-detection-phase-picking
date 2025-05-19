import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import sqrt


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        if attention_dropout:
            self.dropout = nn.Dropout(attention_dropout)
        else:
            self.dropout = None

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.dropout:
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
        else:
            A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)


class Cross_dim_AttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, batch, factor, length, n_heads=4, dropout=None):
        super(Cross_dim_AttentionLayer, self).__init__()
        self.time_attention = AttentionLayer(length, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(length, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(length, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(batch, factor, length))

    def forward(self, x):
        # Cross Time Stage: Directly apply MSA to each dimension
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        x = x.permute(0, 2, 1)
        # Pass through dim_sender
        dim_buffer = self.dim_sender(self.router, x, x)
        # Pass through dim_receiver
        dim_receive = self.dim_receiver(x, dim_buffer, dim_buffer)
        dim_receive = dim_receive.permute(0, 2, 1)
        return dim_receive
